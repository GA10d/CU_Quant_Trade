from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from itertools import permutations, product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_cache import DEFAULT_MODELS_DIR, run_experiment_with_cache
from experiment_presets import (
    build_architecture_runners,
    build_default_experiment_setups,
)
from encoder_only_transformer import load_market_and_macro, split_ordered_train_random_holdout_indices


# ============================================================
# Config
# ============================================================

@dataclass
class TrackBBacktestConfig:
    tradable_assets: tuple[str, ...] = ("SPY", "TLT", "GLD")
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001
    risk_free_rate: float = 0.02
    annualization_factor: int = 252
    execution_lag: int = 1
    validation_ratio: float = 0.20
    test_ratio: float = 0.10
    objective: str = "sharpe"
    allow_action_reuse: bool = False
    fallback_action_name: str | None = None
    split_random_state: int = 42

    @property
    def train_ratio(self) -> float:
        ratio = 1.0 - self.validation_ratio - self.test_ratio
        if ratio <= 0:
            raise ValueError(
                "validation_ratio + test_ratio must be < 1.0, "
                f"got {self.validation_ratio + self.test_ratio:.3f}"
            )
        return ratio


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Track B backtest.")
    parser.add_argument("--cluster-count", type=int, default=4, choices=[3, 4])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--objective", type=str, default="sharpe")
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    parser.add_argument("--execution-lag", type=int, default=1)
    parser.add_argument("--allow-action-reuse", action="store_true", default=False)
    parser.add_argument("--no-action-reuse", dest="allow_action_reuse", action="store_false")
    parser.add_argument("--assets", nargs="+", default=["SPY", "TLT", "GLD"])
    parser.add_argument("--architectures", nargs="*", default=None)
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


# ============================================================
# Action library
# ============================================================

def build_default_action_library(
    n_actions: int,
    tradable_assets: tuple[str, ...] = ("SPY", "TLT", "GLD"),
) -> dict[str, pd.Series]:
    if n_actions == 3:
        templates = {
            "defensive": {"SPY": 0.20, "TLT": 0.55, "GLD": 0.25},
            "balanced": {"SPY": 0.40, "TLT": 0.35, "GLD": 0.25},
            "aggressive": {"SPY": 0.70, "TLT": 0.20, "GLD": 0.10},
        }
    elif n_actions == 4:
        templates = {
            "defensive": {"SPY": 0.20, "TLT": 0.50, "GLD": 0.30},
            "cautious": {"SPY": 0.30, "TLT": 0.40, "GLD": 0.30},
            "rebound": {"SPY": 0.50, "TLT": 0.30, "GLD": 0.20},
            "aggressive": {"SPY": 0.70, "TLT": 0.20, "GLD": 0.10},
        }
    else:
        raise ValueError("Only 3-action or 4-action library is supported.")

    action_library: dict[str, pd.Series] = {}
    for action_name, weights in templates.items():
        w = pd.Series(weights, dtype=float).reindex(tradable_assets).fillna(0.0)
        total = float(w.sum())
        if total <= 0:
            raise ValueError(f"Action {action_name} has no overlap with tradable_assets={tradable_assets}.")
        action_library[action_name] = w / total
    return action_library


def resolve_fallback_action_name(
    action_library: dict[str, pd.Series],
    explicit_name: str | None,
) -> str:
    if explicit_name is not None:
        if explicit_name not in action_library:
            raise ValueError(f"fallback_action_name={explicit_name!r} not found in action library.")
        return explicit_name

    for name in ("balanced", "cautious", "rebound", "defensive", "aggressive"):
        if name in action_library:
            return name
    return next(iter(action_library))


# ============================================================
# Data loading
# ============================================================

def load_tradable_returns(data_config, tradable_assets: tuple[str, ...]) -> pd.DataFrame:
    market_data, _ = load_market_and_macro(data_config)
    missing_assets = [asset for asset in tradable_assets if asset not in market_data.columns]
    if missing_assets:
        raise ValueError(f"Missing tradable assets in market data: {missing_assets}")

    returns = market_data.loc[:, list(tradable_assets)].pct_change(fill_method=None)
    returns = returns.dropna(how="all")
    returns.index.name = "Date"
    return returns


# ============================================================
# Splits
# ============================================================

def split_window_dates(
    window_end_dates: pd.Index,
    validation_ratio: float,
    test_ratio: float,
    random_state: int = 42,
) -> dict[str, pd.Index]:
    n_obs = len(window_end_dates)
    if n_obs < 10:
        raise ValueError(f"Need at least 10 windows, got {n_obs}")

    train_ratio = 1.0 - validation_ratio - test_ratio
    split_indices = split_ordered_train_random_holdout_indices(
        n_obs=n_obs,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    return {
        "train": pd.Index(window_end_dates[split_indices["train"]], name="Date"),
        "validation": pd.Index(window_end_dates[split_indices["validation"]], name="Date"),
        "test": pd.Index(window_end_dates[split_indices["test"]], name="Date"),
    }


def summarize_split_dates(splits: dict[str, pd.Index]) -> pd.DataFrame:
    rows = []
    for split_name, split_dates in splits.items():
        rows.append({
            "split": split_name,
            "n_windows": int(len(split_dates)),
            "start_date": pd.Timestamp(split_dates[0]) if len(split_dates) else pd.NaT,
            "end_date": pd.Timestamp(split_dates[-1]) if len(split_dates) else pd.NaT,
        })
    return pd.DataFrame(rows)


# ============================================================
# Mapping enumeration
# ============================================================

def enumerate_cluster_action_mappings(
    cluster_ids: Iterable[int],
    action_names: Iterable[str],
    allow_action_reuse: bool,
) -> list[dict[int, str]]:
    cluster_list = [int(c) for c in sorted(set(cluster_ids))]
    action_list = list(action_names)

    if not cluster_list:
        return []

    if allow_action_reuse:
        iterator = product(action_list, repeat=len(cluster_list))
    else:
        if len(action_list) < len(cluster_list):
            raise ValueError(
                f"Need at least as many actions as clusters. "
                f"Got {len(action_list)} actions and {len(cluster_list)} clusters."
            )
        iterator = permutations(action_list, len(cluster_list))

    return [
        {cluster_id: action_name for cluster_id, action_name in zip(cluster_list, actions)}
        for actions in iterator
    ]


def serialize_mapping(mapping: dict[int, str]) -> str:
    return "; ".join(f"{cluster}->{action}" for cluster, action in sorted(mapping.items()))


# ============================================================
# Weights construction
# ============================================================

def build_weight_frame_from_mapping(
    cluster_series: pd.Series,
    action_library: dict[str, pd.Series],
    mapping: dict[int, str],
    fallback_action_name: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if cluster_series.empty:
        raise ValueError("cluster_series is empty.")

    assets = list(next(iter(action_library.values())).index)
    fallback_name = resolve_fallback_action_name(action_library, fallback_action_name)

    action_series = cluster_series.map(mapping).fillna(fallback_name)
    weights = pd.DataFrame(index=cluster_series.index, columns=assets, dtype=float)

    for action_name, action_weights in action_library.items():
        mask = action_series == action_name
        if mask.any():
            weights.loc[mask, assets] = action_weights.values

    weights = weights.fillna(0.0)
    row_sums = weights.sum(axis=1)
    valid = row_sums > 0
    weights.loc[valid] = weights.loc[valid].div(row_sums[valid], axis=0)
    return weights, action_series.rename("action")


# ============================================================
# Backtest
# ============================================================

def run_backtest(
    cluster_series: pd.Series,
    asset_returns: pd.DataFrame,
    action_library: dict[str, pd.Series],
    mapping: dict[int, str],
    config: TrackBBacktestConfig,
) -> dict:
    common_dates = cluster_series.index.intersection(asset_returns.index)
    if common_dates.empty:
        raise ValueError("No overlapping dates between cluster labels and asset returns.")

    aligned_clusters = cluster_series.loc[common_dates].copy()
    aligned_returns = asset_returns.loc[common_dates, list(config.tradable_assets)].copy()
    aligned_returns = aligned_returns.dropna(how="any")
    aligned_clusters = aligned_clusters.loc[aligned_returns.index]

    target_weights, action_series = build_weight_frame_from_mapping(
        cluster_series=aligned_clusters,
        action_library=action_library,
        mapping=mapping,
        fallback_action_name=config.fallback_action_name,
    )

    executed_weights = target_weights.shift(config.execution_lag).fillna(0.0)

    turnover = executed_weights.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = executed_weights.iloc[0].abs().sum()

    gross_returns = (executed_weights * aligned_returns).sum(axis=1)
    transaction_costs = turnover * config.transaction_cost
    portfolio_returns = gross_returns - transaction_costs
    portfolio_nav = (1.0 + portfolio_returns).cumprod() * config.initial_capital

    return {
        "cluster_series": aligned_clusters.rename("cluster"),
        "action_series": action_series,
        "target_weights": target_weights,
        "executed_weights": executed_weights,
        "asset_returns": aligned_returns,
        "gross_returns": gross_returns.rename("gross_return"),
        "transaction_costs": transaction_costs.rename("transaction_cost"),
        "portfolio_returns": portfolio_returns.rename("portfolio_return"),
        "portfolio_nav": portfolio_nav.rename("portfolio_nav"),
        "turnover": turnover.rename("turnover"),
    }


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    portfolio_returns: pd.Series,
    turnover: pd.Series,
    config: TrackBBacktestConfig,
) -> dict:
    returns = portfolio_returns.dropna()
    turnover = turnover.loc[returns.index].fillna(0.0)

    if returns.empty:
        return {
            "n_days": 0,
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "final_nav": np.nan,
            "win_rate": np.nan,
            "avg_turnover": np.nan,
            "total_turnover": np.nan,
        }

    cumulative = (1.0 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0)
    annual_return = float(cumulative.iloc[-1] ** (config.annualization_factor / len(returns)) - 1.0)
    annual_vol = float(returns.std(ddof=0) * np.sqrt(config.annualization_factor))

    if annual_vol > 0:
        sharpe = float((annual_return - config.risk_free_rate) / annual_vol)
    else:
        sharpe = np.nan

    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=0) * np.sqrt(config.annualization_factor)) if len(downside) > 0 else np.nan
    if downside_vol and np.isfinite(downside_vol) and downside_vol > 0:
        sortino = float((annual_return - config.risk_free_rate) / downside_vol)
    else:
        sortino = np.nan

    drawdown = cumulative / cumulative.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else np.nan

    if np.isfinite(max_drawdown) and max_drawdown < 0:
        calmar = float(annual_return / abs(max_drawdown))
    else:
        calmar = np.nan

    final_nav = float(config.initial_capital * cumulative.iloc[-1])
    win_rate = float((returns > 0).mean())

    return {
        "n_days": int(len(returns)),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "final_nav": final_nav,
        "win_rate": win_rate,
        "avg_turnover": float(turnover.mean()),
        "total_turnover": float(turnover.sum()),
    }


def metric_sort_direction(metric_name: str) -> tuple[str, bool]:
    descending_metrics = {"total_return", "annual_return", "sharpe", "sortino", "calmar", "final_nav", "win_rate"}
    ascending_metrics = {"annual_vol", "max_drawdown", "avg_turnover", "total_turnover"}

    if metric_name in descending_metrics:
        return metric_name, True
    if metric_name in ascending_metrics:
        return metric_name, False
    raise ValueError(f"Unsupported objective metric: {metric_name}")


def objective_value(metrics: dict, objective: str) -> float:
    value = metrics.get(objective, np.nan)
    _, descending = metric_sort_direction(objective)

    if value is None or not np.isfinite(value):
        return -np.inf if descending else np.inf

    return float(value) if descending else -float(value)


# ============================================================
# Search best mapping
# ============================================================

def search_best_mapping(
    cluster_series: pd.Series,
    asset_returns: pd.DataFrame,
    action_library: dict[str, pd.Series],
    validation_dates: pd.Index,
    config: TrackBBacktestConfig,
) -> dict:
    validation_clusters = cluster_series.loc[cluster_series.index.intersection(validation_dates)].dropna()
    if validation_clusters.empty:
        raise ValueError("Validation cluster series is empty after alignment.")

    candidate_mappings = enumerate_cluster_action_mappings(
        cluster_ids=cluster_series.dropna().unique(),
        action_names=action_library.keys(),
        allow_action_reuse=config.allow_action_reuse,
    )
    if not candidate_mappings:
        raise ValueError("No candidate mappings were generated.")

    search_rows = []
    best_mapping = None
    best_score = -np.inf
    best_backtest = None
    best_metrics = None

    for mapping in candidate_mappings:
        bt = run_backtest(
            cluster_series=cluster_series,
            asset_returns=asset_returns,
            action_library=action_library,
            mapping=mapping,
            config=config,
        )
        validation_returns = bt["portfolio_returns"].loc[bt["portfolio_returns"].index.intersection(validation_dates)]
        validation_turnover = bt["turnover"].loc[validation_returns.index]
        metrics = calculate_metrics(validation_returns, validation_turnover, config)
        score = objective_value(metrics, config.objective)

        row = {
            "mapping": serialize_mapping(mapping),
            "objective_value": score,
        }
        row.update({f"validation_{k}": v for k, v in metrics.items()})
        search_rows.append(row)

        if score > best_score:
            best_score = score
            best_mapping = mapping
            best_backtest = bt
            best_metrics = metrics

    search_df = pd.DataFrame(search_rows).sort_values("objective_value", ascending=False).reset_index(drop=True)

    return {
        "best_mapping": best_mapping,
        "best_mapping_text": serialize_mapping(best_mapping),
        "best_score": best_score,
        "validation_metrics": best_metrics,
        "search_results": search_df,
        "backtest_result": best_backtest,
    }


# ============================================================
# Single experiment evaluation
# ============================================================

def evaluate_single_experiment(
    experiment_result: dict,
    asset_returns: pd.DataFrame,
    action_library: dict[str, pd.Series],
    config: TrackBBacktestConfig,
) -> dict:
    window_end_dates = pd.Index(experiment_result["window_end_dates"], name="Date")
    cluster_series = pd.Series(
        experiment_result["cluster_labels"],
        index=window_end_dates,
        name="cluster"
    )

    splits = split_window_dates(
        window_end_dates=window_end_dates,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        random_state=config.split_random_state,
    )
    split_summary = summarize_split_dates(splits)

    search_result = search_best_mapping(
        cluster_series=cluster_series,
        asset_returns=asset_returns,
        action_library=action_library,
        validation_dates=splits["validation"],
        config=config,
    )
    best_backtest = search_result["backtest_result"]

    validation_returns = best_backtest["portfolio_returns"].loc[
        best_backtest["portfolio_returns"].index.intersection(splits["validation"])
    ]
    validation_turnover = best_backtest["turnover"].loc[validation_returns.index]
    validation_metrics = calculate_metrics(validation_returns, validation_turnover, config)

    test_returns = best_backtest["portfolio_returns"].loc[
        best_backtest["portfolio_returns"].index.intersection(splits["test"])
    ]
    test_turnover = best_backtest["turnover"].loc[test_returns.index]
    test_metrics = calculate_metrics(test_returns, test_turnover, config)

    full_returns = best_backtest["portfolio_returns"]
    full_turnover = best_backtest["turnover"]
    full_metrics = calculate_metrics(full_returns, full_turnover, config)

    summary = {
        "experiment_name": experiment_result["experiment_name"],
        "architecture": experiment_result["summary"].get("architecture", experiment_result["experiment_name"]),
        "n_clusters": int(np.unique(cluster_series.dropna()).size),
        "best_mapping": search_result["best_mapping_text"],
        "objective": config.objective,
    }
    summary.update({f"validation_{k}": v for k, v in validation_metrics.items()})
    summary.update({f"test_{k}": v for k, v in test_metrics.items()})
    summary.update({f"full_{k}": v for k, v in full_metrics.items()})

    return {
        "summary": summary,
        "splits": splits,
        "split_summary": split_summary,
        "best_mapping": search_result["best_mapping"],
        "best_mapping_text": search_result["best_mapping_text"],
        "search_results": search_result["search_results"],
        "backtest_result": best_backtest,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "full_metrics": full_metrics,
    }


# ============================================================
# Ranking
# ============================================================

def build_ranking_tables(
    evaluation_results: list[dict],
    objective: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_df = pd.DataFrame([result["summary"] for result in evaluation_results])
    if summary_df.empty:
        return summary_df, summary_df, summary_df

    _, descending = metric_sort_direction(objective)
    validation_col = f"validation_{objective}"
    test_col = f"test_{objective}"

    validation_ranking = summary_df.sort_values(validation_col, ascending=not descending).reset_index(drop=True)
    validation_ranking.insert(0, "validation_rank", np.arange(1, len(validation_ranking) + 1))

    test_ranking = summary_df.sort_values(test_col, ascending=not descending).reset_index(drop=True)
    test_ranking.insert(0, "test_rank", np.arange(1, len(test_ranking) + 1))

    overall = validation_ranking.merge(
        test_ranking[["experiment_name", "test_rank"]],
        on="experiment_name",
        how="left",
    )
    overall["avg_rank"] = (overall["validation_rank"] + overall["test_rank"]) / 2.0
    overall = overall.sort_values(["avg_rank", "test_rank", "validation_rank"]).reset_index(drop=True)
    overall.insert(0, "overall_rank", np.arange(1, len(overall) + 1))

    return validation_ranking, test_ranking, overall


# ============================================================
# Save outputs
# ============================================================

def save_nav_plot(nav_frame: pd.DataFrame, output_path: Path, title: str) -> None:
    if nav_frame.empty:
        return

    plt.figure(figsize=(14, 7))
    for column in nav_frame.columns:
        plt.plot(nav_frame.index, nav_frame[column], label=column, linewidth=1.6)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized NAV")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_nav_frame(evaluation_results: list[dict], split_name: str | None = None) -> pd.DataFrame:
    nav_series = {}
    for result in evaluation_results:
        name = result["summary"]["experiment_name"]
        portfolio_returns = result["backtest_result"]["portfolio_returns"].copy()
        if split_name is not None:
            split_dates = result["splits"][split_name]
            portfolio_returns = portfolio_returns.loc[portfolio_returns.index.intersection(split_dates)]

        nav = (1.0 + portfolio_returns.dropna()).cumprod()
        nav.name = name
        nav_series[name] = nav

    if not nav_series:
        return pd.DataFrame()

    nav_frame = pd.concat(nav_series, axis=1).sort_index()
    return nav_frame.ffill()


def save_single_experiment_outputs(output_dir: Path, result: dict) -> None:
    summary = result["summary"]
    experiment_name = summary["experiment_name"]
    safe_name = experiment_name.replace("/", "_")

    exp_dir = output_dir / safe_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([summary]).to_csv(exp_dir / "summary.csv", index=False)
    result["split_summary"].to_csv(exp_dir / "split_summary.csv", index=False)
    result["search_results"].to_csv(exp_dir / "mapping_search.csv", index=False)

    daily_backtest = pd.concat(
        [
            result["backtest_result"]["cluster_series"],
            result["backtest_result"]["action_series"],
            result["backtest_result"]["portfolio_returns"],
            result["backtest_result"]["transaction_costs"],
            result["backtest_result"]["turnover"],
            result["backtest_result"]["portfolio_nav"],
        ],
        axis=1,
    )
    daily_backtest.to_csv(exp_dir / "daily_backtest.csv")

    result["backtest_result"]["target_weights"].to_csv(exp_dir / "target_weights.csv")
    result["backtest_result"]["executed_weights"].to_csv(exp_dir / "executed_weights.csv")

    nav = (1.0 + result["backtest_result"]["portfolio_returns"].dropna()).cumprod()
    if not nav.empty:
        plt.figure(figsize=(12, 5))
        plt.plot(nav.index, nav.values, linewidth=1.8)
        plt.title(f"{experiment_name} NAV")
        plt.xlabel("Date")
        plt.ylabel("Normalized NAV")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(exp_dir / "nav.png", dpi=150, bbox_inches="tight")
        plt.close()


def save_all_outputs(
    output_dir: Path,
    evaluation_results: list[dict],
    validation_ranking: pd.DataFrame,
    test_ranking: pd.DataFrame,
    overall_ranking: pd.DataFrame,
    action_library: dict[str, pd.Series],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    action_library_df = pd.DataFrame(action_library).T
    action_library_df.index.name = "action"
    action_library_df.to_csv(output_dir / "action_library.csv")

    validation_ranking.to_csv(output_dir / "validation_ranking.csv", index=False)
    test_ranking.to_csv(output_dir / "test_ranking.csv", index=False)
    overall_ranking.to_csv(output_dir / "overall_ranking.csv", index=False)

    if evaluation_results:
        evaluation_results[0]["split_summary"].to_csv(output_dir / "global_split_summary.csv", index=False)

    for result in evaluation_results:
        save_single_experiment_outputs(output_dir, result)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    full_nav = build_nav_frame(evaluation_results, split_name=None)
    validation_nav = build_nav_frame(evaluation_results, split_name="validation")
    test_nav = build_nav_frame(evaluation_results, split_name="test")

    save_nav_plot(full_nav, plots_dir / "full_period_nav.png", "Track B Full-Period NAV Comparison")
    save_nav_plot(validation_nav, plots_dir / "validation_nav.png", "Track B Validation NAV Comparison")
    save_nav_plot(test_nav, plots_dir / "test_nav.png", "Track B Test NAV Comparison")


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    track_b_dir = Path(__file__).resolve().parent
    data_dir = track_b_dir.parent / "data"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else track_b_dir.parent / "artifacts" / f"track_b_backtest_k{args.cluster_count}"
    )

    config = TrackBBacktestConfig(
        tradable_assets=tuple(args.assets),
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        risk_free_rate=args.risk_free_rate,
        execution_lag=args.execution_lag,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        objective=args.objective,
        allow_action_reuse=args.allow_action_reuse,
    )

    action_library = build_default_action_library(
        n_actions=args.cluster_count,
        tradable_assets=config.tradable_assets,
    )

    print("=" * 70)
    print("Track B Standalone Backtest")
    print("=" * 70)
    print("Action library:")
    print(pd.DataFrame(action_library).T.to_string(float_format=lambda x: f"{x:.2f}"))
    print()

    runners = build_architecture_runners()
    experiment_setups = build_default_experiment_setups(
        data_dir=data_dir,
        hmm_enabled=False,
        target_cluster_count=args.cluster_count,
        device=args.device,
    )

    if args.architectures:
        requested = set(args.architectures)
        experiment_setups = [setup for setup in experiment_setups if setup["architecture"] in requested]

    if not experiment_setups:
        raise ValueError("No experiment setups selected. Check --architectures.")

    experiment_setups = [
        {
            **setup,
            "training_config": replace(
                setup["training_config"],
                train_ratio=config.train_ratio,
                validation_ratio=args.validation_ratio,
                test_ratio=args.test_ratio,
                device=args.device,
            ),
        }
        for setup in experiment_setups
    ]

    returns_data_config = experiment_setups[0]["data_config"]
    asset_returns = load_tradable_returns(
        data_config=returns_data_config,
        tradable_assets=config.tradable_assets,
    )

    print("Selected models:")
    for setup in experiment_setups:
        print(f"  - {setup['name']}")
    print()

    evaluation_results = []

    for setup in experiment_setups:
        print(f"=== Running {setup['name']} ===")
        runner = runners[setup["architecture"]]

        experiment_result = run_experiment_with_cache(
            runner=runner,
            experiment_name=setup["name"],
            data_config=setup["data_config"],
            model_config=setup["model_config"],
            training_config=setup["training_config"],
            clustering_config=setup["clustering_config"],
            hmm_config=setup["hmm_config"],
            models_dir=args.models_dir,
            verbose=True,
        )

        evaluation_result = evaluate_single_experiment(
            experiment_result=experiment_result,
            asset_returns=asset_returns,
            action_library=action_library,
            config=config,
        )
        evaluation_results.append(evaluation_result)

        summary = evaluation_result["summary"]
        print(
            f"best_mapping={summary['best_mapping']} | "
            f"validation_{args.objective}={summary[f'validation_{args.objective}']:.4f} | "
            f"test_{args.objective}={summary[f'test_{args.objective}']:.4f} | "
            f"full_{args.objective}={summary[f'full_{args.objective}']:.4f}"
        )
        print()

    validation_ranking, test_ranking, overall_ranking = build_ranking_tables(
        evaluation_results=evaluation_results,
        objective=args.objective,
    )

    save_all_outputs(
        output_dir=output_dir,
        evaluation_results=evaluation_results,
        validation_ranking=validation_ranking,
        test_ranking=test_ranking,
        overall_ranking=overall_ranking,
        action_library=action_library,
    )

    validation_cols = [
        "validation_rank",
        "experiment_name",
        "architecture",
        f"validation_{args.objective}",
        "validation_annual_return",
        "validation_max_drawdown",
        "best_mapping",
    ]
    test_cols = [
        "test_rank",
        "experiment_name",
        "architecture",
        f"test_{args.objective}",
        "test_annual_return",
        "test_max_drawdown",
        "best_mapping",
    ]
    overall_cols = [
        "overall_rank",
        "experiment_name",
        "architecture",
        "validation_rank",
        "test_rank",
        "avg_rank",
        f"validation_{args.objective}",
        f"test_{args.objective}",
    ]

    print("=" * 70)
    print("Validation Ranking")
    print("=" * 70)
    print(validation_ranking[validation_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    print("=" * 70)
    print("Test Ranking")
    print("=" * 70)
    print(test_ranking[test_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    print("=" * 70)
    print("Overall Ranking")
    print("=" * 70)
    print(overall_ranking[overall_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()