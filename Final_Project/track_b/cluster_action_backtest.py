from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from encoder_only_transformer import (
    HMMLEARN_AVAILABLE,
    HMMReferenceConfig,
    SequenceDataConfig,
    TrainingConfig,
    build_hmm_reference,
    load_market_and_macro,
    split_ordered_train_random_holdout_indices,
)
from experiment_utils import prepare_sequence_experiment_inputs


@dataclass
class ActionBacktestConfig:
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
                "validation_ratio + test_ratio must be < 1.0 "
                f"(got {self.validation_ratio + self.test_ratio:.3f})."
            )
        return ratio


def build_default_action_library(
    n_actions: int,
    tradable_assets: tuple[str, ...] = ("SPY", "TLT", "GLD"),
) -> dict[str, pd.Series]:
    if n_actions == 3:
        base_templates = {
            "defensive": {"SPY": 0.20, "TLT": 0.55, "GLD": 0.25},
            "balanced": {"SPY": 0.40, "TLT": 0.35, "GLD": 0.25},
            "aggressive": {"SPY": 0.70, "TLT": 0.20, "GLD": 0.10},
        }
    elif n_actions == 4:
        base_templates = {
            "defensive": {"SPY": 0.20, "TLT": 0.50, "GLD": 0.30},
            "cautious": {"SPY": 0.30, "TLT": 0.40, "GLD": 0.30},
            "rebound": {"SPY": 0.50, "TLT": 0.30, "GLD": 0.20},
            "aggressive": {"SPY": 0.70, "TLT": 0.20, "GLD": 0.10},
        }
    else:
        raise ValueError("Default action library only supports 3-action or 4-action configurations.")

    action_library: dict[str, pd.Series] = {}
    for action_name, weights in base_templates.items():
        action_weights = pd.Series(weights, dtype=float).reindex(tradable_assets).fillna(0.0)
        total_weight = float(action_weights.sum())
        if total_weight <= 0:
            raise ValueError(f"Action template {action_name} has no overlap with tradable_assets={tradable_assets}.")
        action_library[action_name] = action_weights / total_weight
    return action_library


def load_tradable_returns(
    data_config: SequenceDataConfig,
    tradable_assets: tuple[str, ...],
) -> pd.DataFrame:
    market_data, _ = load_market_and_macro(data_config)
    missing_assets = [asset for asset in tradable_assets if asset not in market_data.columns]
    if missing_assets:
        raise ValueError(f"Missing tradable assets in market data: {', '.join(missing_assets)}")

    returns = market_data.loc[:, list(tradable_assets)].pct_change(fill_method=None)
    returns = returns.dropna(how="all")
    returns.index.name = "Date"
    return returns


def build_hmm_baseline_experiment_result(
    data_config: SequenceDataConfig,
    target_cluster_count: int,
    random_state: int = 42,
    covariance_type: str = "diag",
    n_restarts: int = 6,
    n_iter: int = 500,
    experiment_name: str = "hmm_baseline",
) -> dict | None:
    if not HMMLEARN_AVAILABLE:
        return None

    prepared_inputs = prepare_sequence_experiment_inputs(data_config)
    hmm_config = HMMReferenceConfig(
        enabled=True,
        state_candidates=(target_cluster_count,),
        covariance_type=covariance_type,
        n_restarts=n_restarts,
        n_iter=n_iter,
    )
    hmm_results = build_hmm_reference(
        data_config=data_config,
        market_data=prepared_inputs["market_data"],
        macro_data=prepared_inputs["macro_data"],
        hmm_config=hmm_config,
        random_state=random_state,
    )

    hmm_reference = hmm_results["hmm_reference"]
    aligned_dates = pd.Index(prepared_inputs["window_end_dates"], name="Date").intersection(hmm_reference.index)
    aligned_dates = aligned_dates.sort_values()
    if aligned_dates.empty:
        raise ValueError("No overlapping dates found between HMM reference and Track B windows.")

    positions = pd.Index(prepared_inputs["window_end_dates"]).get_indexer(aligned_dates)
    if np.any(positions < 0):
        raise ValueError("Failed to align HMM baseline to Track B window_end_dates.")

    cluster_labels = hmm_reference.loc[aligned_dates, "hmm_state"].astype(int).to_numpy()
    numeric_hmm_columns = [
        column
        for column in hmm_reference.columns
        if column not in {"hmm_state", "hmm_regime"} and pd.api.types.is_numeric_dtype(hmm_reference[column])
    ]
    embeddings = hmm_reference.loc[aligned_dates, numeric_hmm_columns].to_numpy()
    cluster_summary = (
        pd.DataFrame({"Date": aligned_dates, "cluster": cluster_labels})
        .groupby("cluster")
        .agg(
            n_windows=("cluster", "size"),
            dominant_hmm_regime=("Date", lambda dates: hmm_reference.loc[list(dates), "hmm_regime"].value_counts().index[0]),
        )
        .reset_index()
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    comparison_table = pd.crosstab(
        hmm_reference.loc[aligned_dates, "hmm_state"],
        hmm_reference.loc[aligned_dates, "hmm_regime"],
    )

    summary = {
        "experiment_name": experiment_name,
        "architecture": "hmm_baseline",
        "input_dim": int(len(numeric_hmm_columns)),
        "window_size": int(data_config.window_size),
        "n_sequence_rows": int(len(prepared_inputs["sequence_panel"])),
        "n_windows": int(len(aligned_dates)),
        "used_sequence_features": ", ".join(prepared_inputs["sequence_metadata"]["used_features"]),
        "skipped_sequence_features": ", ".join(prepared_inputs["sequence_metadata"]["skipped_features"]),
        "best_epoch": 0,
        "best_val_loss": np.nan,
        "target_cluster_count": int(target_cluster_count),
        "silhouette": np.nan,
        "ari_vs_hmm": 1.0,
        "nmi_vs_hmm": 1.0,
    }

    return {
        "experiment_name": experiment_name,
        "summary": summary,
        "data_config": data_config,
        "model_config": None,
        "training_config": TrainingConfig(random_state=random_state),
        "clustering_config": None,
        "hmm_config": hmm_config,
        "market_data": prepared_inputs["market_data"],
        "macro_data": prepared_inputs["macro_data"],
        "sequence_panel": prepared_inputs["sequence_panel"],
        "sequence_metadata": prepared_inputs["sequence_metadata"],
        "sequence_scaler": prepared_inputs["sequence_scaler"],
        "windows": prepared_inputs["windows"][positions],
        "window_end_dates": aligned_dates,
        "model": hmm_results["hmm_model"],
        "history_df": pd.DataFrame(),
        "embeddings": embeddings,
        "cluster_scan": pd.DataFrame([{"n_clusters": int(target_cluster_count), "silhouette": np.nan}]),
        "cluster_model": hmm_results["hmm_model"],
        "cluster_labels": cluster_labels,
        "cluster_summary": cluster_summary,
        "comparison_table": comparison_table,
        "hmm_results": hmm_results,
        "device": "cpu",
    }


def split_window_dates(
    window_end_dates: pd.Index,
    validation_ratio: float,
    test_ratio: float,
    random_state: int = 42,
) -> dict[str, pd.Index]:
    n_obs = len(window_end_dates)
    if n_obs < 10:
        raise ValueError(f"Need at least 10 windows for backtesting splits, got {n_obs}.")

    train_ratio = 1.0 - validation_ratio - test_ratio
    split_indices = split_ordered_train_random_holdout_indices(
        n_obs=n_obs,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    train_dates = window_end_dates[split_indices["train"]]
    validation_dates = window_end_dates[split_indices["validation"]]
    test_dates = window_end_dates[split_indices["test"]]

    return {
        "train": pd.Index(train_dates, name="Date"),
        "validation": pd.Index(validation_dates, name="Date"),
        "test": pd.Index(test_dates, name="Date"),
    }


def summarize_split_dates(splits: dict[str, pd.Index]) -> pd.DataFrame:
    rows = []
    for split_name, split_dates in splits.items():
        if len(split_dates) == 0:
            rows.append(
                {
                    "split": split_name,
                    "n_windows": 0,
                    "start_date": pd.NaT,
                    "end_date": pd.NaT,
                    "selection": "empty",
                }
            )
        else:
            rows.append(
                {
                    "split": split_name,
                    "n_windows": int(len(split_dates)),
                    "start_date": pd.Timestamp(split_dates[0]),
                    "end_date": pd.Timestamp(split_dates[-1]),
                    "selection": "ordered_oldest" if split_name == "train" else "random_from_holdout",
                }
            )
    return pd.DataFrame(rows)


def _fallback_action_name(
    action_library: dict[str, pd.Series],
    explicit_name: str | None,
) -> str:
    if explicit_name is not None:
        if explicit_name not in action_library:
            raise ValueError(f"fallback_action_name={explicit_name!r} is not in the action library.")
        return explicit_name

    for preferred in ("balanced", "cautious", "rebound", "defensive", "aggressive"):
        if preferred in action_library:
            return preferred
    return next(iter(action_library))


def enumerate_cluster_action_mappings(
    cluster_ids: Iterable[int],
    action_names: Iterable[str],
    allow_action_reuse: bool,
) -> list[dict[int, str]]:
    cluster_list = [int(cluster_id) for cluster_id in sorted(set(cluster_ids))]
    action_list = list(action_names)
    if not cluster_list:
        return []

    mappings: list[dict[int, str]] = []
    if allow_action_reuse:
        iterator = product(action_list, repeat=len(cluster_list))
    else:
        if len(action_list) < len(cluster_list):
            raise ValueError(
                "Cannot enumerate one-to-one mappings when there are fewer actions than clusters: "
                f"{len(action_list)} actions vs {len(cluster_list)} clusters."
            )
        iterator = permutations(action_list, len(cluster_list))

    for actions in iterator:
        mappings.append({cluster_id: action_name for cluster_id, action_name in zip(cluster_list, actions)})
    return mappings


def serialize_mapping(mapping: dict[int, str]) -> str:
    return "; ".join(f"{cluster}->{action}" for cluster, action in sorted(mapping.items()))


def build_weight_frame_from_mapping(
    cluster_series: pd.Series,
    action_library: dict[str, pd.Series],
    mapping: dict[int, str],
    fallback_action_name: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if cluster_series.empty:
        raise ValueError("cluster_series is empty.")

    assets = list(next(iter(action_library.values())).index)
    fallback_name = _fallback_action_name(action_library, fallback_action_name)
    action_series = cluster_series.map(mapping).fillna(fallback_name)

    weights = pd.DataFrame(index=cluster_series.index, columns=assets, dtype=float)
    for action_name, action_weights in action_library.items():
        mask = action_series == action_name
        if mask.any():
            weights.loc[mask, assets] = action_weights.values

    weights = weights.fillna(0.0)
    row_sums = weights.sum(axis=1)
    valid_rows = row_sums > 0
    weights.loc[valid_rows] = weights.loc[valid_rows].div(row_sums[valid_rows], axis=0)
    return weights, action_series.rename("action")


def run_action_mapping_backtest(
    cluster_series: pd.Series,
    asset_returns: pd.DataFrame,
    action_library: dict[str, pd.Series],
    mapping: dict[int, str],
    config: ActionBacktestConfig,
) -> dict:
    common_dates = cluster_series.index.intersection(asset_returns.index)
    if common_dates.empty:
        raise ValueError("No overlapping dates between cluster_series and asset_returns.")

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
        "cluster_series": aligned_clusters,
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


def calculate_backtest_metrics(
    portfolio_returns: pd.Series,
    turnover: pd.Series,
    config: ActionBacktestConfig,
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
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "final_nav": np.nan,
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

    drawdown = cumulative / cumulative.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else np.nan
    if np.isfinite(max_drawdown) and max_drawdown < 0:
        calmar = float(annual_return / abs(max_drawdown))
    else:
        calmar = np.nan

    final_nav = float(config.initial_capital * cumulative.iloc[-1])
    return {
        "n_days": int(len(returns)),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "final_nav": final_nav,
        "avg_turnover": float(turnover.mean()),
        "total_turnover": float(turnover.sum()),
    }


def _metric_sort_key(metric_name: str) -> tuple[str, bool]:
    descending_metrics = {"total_return", "annual_return", "sharpe", "calmar", "final_nav"}
    ascending_metrics = {"annual_vol", "max_drawdown", "avg_turnover", "total_turnover"}
    if metric_name in descending_metrics:
        return metric_name, True
    if metric_name in ascending_metrics:
        return metric_name, False
    raise ValueError(f"Unsupported objective metric: {metric_name}")


def _objective_value(metrics: dict, objective: str) -> float:
    value = metrics.get(objective, np.nan)
    if value is None or not np.isfinite(value):
        return -np.inf if _metric_sort_key(objective)[1] else np.inf
    if _metric_sort_key(objective)[1]:
        return float(value)
    return -float(value)


def search_best_cluster_action_mapping(
    cluster_series: pd.Series,
    asset_returns: pd.DataFrame,
    action_library: dict[str, pd.Series],
    validation_dates: pd.Index,
    config: ActionBacktestConfig,
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
        raise ValueError("No candidate cluster->action mappings were generated.")

    search_rows: list[dict] = []
    best_mapping = None
    best_score = -np.inf
    best_backtest = None
    best_metrics = None

    for mapping in candidate_mappings:
        backtest_result = run_action_mapping_backtest(
            cluster_series=cluster_series,
            asset_returns=asset_returns,
            action_library=action_library,
            mapping=mapping,
            config=config,
        )
        validation_returns = backtest_result["portfolio_returns"].loc[
            backtest_result["portfolio_returns"].index.intersection(validation_dates)
        ]
        validation_turnover = backtest_result["turnover"].loc[validation_returns.index]
        validation_metrics = calculate_backtest_metrics(validation_returns, validation_turnover, config)
        objective_value = _objective_value(validation_metrics, config.objective)

        row = {
            "mapping": serialize_mapping(mapping),
            "objective_value": objective_value,
        }
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        search_rows.append(row)

        if objective_value > best_score:
            best_score = objective_value
            best_mapping = mapping
            best_backtest = backtest_result
            best_metrics = validation_metrics

    assert best_mapping is not None
    assert best_backtest is not None
    assert best_metrics is not None

    search_df = pd.DataFrame(search_rows).sort_values("objective_value", ascending=False).reset_index(drop=True)
    return {
        "best_mapping": best_mapping,
        "best_mapping_text": serialize_mapping(best_mapping),
        "best_score": best_score,
        "validation_metrics": best_metrics,
        "search_results": search_df,
        "backtest_result": best_backtest,
    }


def evaluate_model_cluster_actions(
    experiment_result: dict,
    asset_returns: pd.DataFrame,
    action_library: dict[str, pd.Series],
    config: ActionBacktestConfig,
) -> dict:
    window_end_dates = pd.Index(experiment_result["window_end_dates"], name="Date")
    cluster_series = pd.Series(experiment_result["cluster_labels"], index=window_end_dates, name="cluster")
    splits = split_window_dates(
        window_end_dates=window_end_dates,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        random_state=config.split_random_state,
    )
    split_summary = summarize_split_dates(splits)

    search_result = search_best_cluster_action_mapping(
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
    validation_metrics = calculate_backtest_metrics(validation_returns, validation_turnover, config)

    test_returns = best_backtest["portfolio_returns"].loc[
        best_backtest["portfolio_returns"].index.intersection(splits["test"])
    ]
    test_turnover = best_backtest["turnover"].loc[test_returns.index]
    test_metrics = calculate_backtest_metrics(test_returns, test_turnover, config)

    summary = {
        "experiment_name": experiment_result["experiment_name"],
        "architecture": experiment_result["summary"].get("architecture"),
        "n_clusters": int(np.unique(cluster_series.values).size),
        "target_cluster_count": int(experiment_result["summary"].get("target_cluster_count", np.unique(cluster_series.values).size)),
        "best_mapping": search_result["best_mapping_text"],
        "objective": config.objective,
    }
    summary.update({f"validation_{key}": value for key, value in validation_metrics.items()})
    summary.update({f"test_{key}": value for key, value in test_metrics.items()})

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
    }


def evaluate_random_choice_baseline(
    data_config: SequenceDataConfig,
    asset_returns: pd.DataFrame,
    action_library: dict[str, pd.Series],
    config: ActionBacktestConfig,
    random_state: int = 42,
    experiment_name: str = "random_choice",
) -> dict:
    prepared_inputs = prepare_sequence_experiment_inputs(data_config)
    window_end_dates = pd.Index(prepared_inputs["window_end_dates"], name="Date")
    splits = split_window_dates(
        window_end_dates=window_end_dates,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        random_state=config.split_random_state,
    )
    split_summary = summarize_split_dates(splits)

    action_names = list(action_library.keys())
    if not action_names:
        raise ValueError("Action library is empty.")

    rng = np.random.default_rng(random_state)
    random_actions = rng.choice(action_names, size=len(window_end_dates), replace=True)
    action_to_code = {action_name: code for code, action_name in enumerate(action_names)}
    cluster_series = pd.Series(
        [action_to_code[action_name] for action_name in random_actions],
        index=window_end_dates,
        name="cluster",
        dtype=int,
    )
    mapping = {code: action_name for action_name, code in action_to_code.items()}
    backtest_result = run_action_mapping_backtest(
        cluster_series=cluster_series,
        asset_returns=asset_returns,
        action_library=action_library,
        mapping=mapping,
        config=config,
    )

    validation_returns = backtest_result["portfolio_returns"].loc[
        backtest_result["portfolio_returns"].index.intersection(splits["validation"])
    ]
    validation_turnover = backtest_result["turnover"].loc[validation_returns.index]
    validation_metrics = calculate_backtest_metrics(validation_returns, validation_turnover, config)

    test_returns = backtest_result["portfolio_returns"].loc[
        backtest_result["portfolio_returns"].index.intersection(splits["test"])
    ]
    test_turnover = backtest_result["turnover"].loc[test_returns.index]
    test_metrics = calculate_backtest_metrics(test_returns, test_turnover, config)

    best_mapping_text = f"random_daily_choice(seed={random_state})"
    search_results = pd.DataFrame(
        [
            {
                "mapping": best_mapping_text,
                "objective_value": _objective_value(validation_metrics, config.objective),
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
        ]
    )
    summary = {
        "experiment_name": experiment_name,
        "architecture": "random_choice",
        "n_clusters": int(len(action_names)),
        "target_cluster_count": int(len(action_names)),
        "best_mapping": best_mapping_text,
        "objective": config.objective,
    }
    summary.update({f"validation_{key}": value for key, value in validation_metrics.items()})
    summary.update({f"test_{key}": value for key, value in test_metrics.items()})

    return {
        "summary": summary,
        "splits": splits,
        "split_summary": split_summary,
        "best_mapping": mapping,
        "best_mapping_text": best_mapping_text,
        "search_results": search_results,
        "backtest_result": backtest_result,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }


def build_model_ranking_tables(
    evaluation_results: list[dict],
    objective: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_df = pd.DataFrame([result["summary"] for result in evaluation_results])
    if summary_df.empty:
        return summary_df, summary_df, summary_df

    _, descending = _metric_sort_key(objective)
    validation_column = f"validation_{objective}"
    test_column = f"test_{objective}"

    validation_ranking = summary_df.sort_values(validation_column, ascending=not descending).reset_index(drop=True)
    validation_ranking.insert(0, "validation_rank", np.arange(1, len(validation_ranking) + 1))

    test_ranking = summary_df.sort_values(test_column, ascending=not descending).reset_index(drop=True)
    test_ranking.insert(0, "test_rank", np.arange(1, len(test_ranking) + 1))

    combined = validation_ranking.merge(
        test_ranking[["experiment_name", "test_rank"]],
        on="experiment_name",
        how="left",
    )
    return validation_ranking, test_ranking, combined


def save_evaluation_artifacts(
    output_dir: str | Path,
    evaluation_results: list[dict],
    validation_ranking: pd.DataFrame,
    test_ranking: pd.DataFrame,
    combined_ranking: pd.DataFrame,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    validation_ranking.to_csv(output_path / "validation_ranking.csv", index=False)
    test_ranking.to_csv(output_path / "test_ranking.csv", index=False)
    combined_ranking.to_csv(output_path / "combined_ranking.csv", index=False)
    if evaluation_results:
        evaluation_results[0]["split_summary"].to_csv(output_path / "split_summary.csv", index=False)

    search_dir = output_path / "mapping_search"
    backtest_dir = output_path / "backtests"
    search_dir.mkdir(exist_ok=True)
    backtest_dir.mkdir(exist_ok=True)

    for result in evaluation_results:
        experiment_name = result["summary"]["experiment_name"]
        safe_name = experiment_name.replace("/", "_")
        result["search_results"].to_csv(search_dir / f"{safe_name}_validation_search.csv", index=False)

        backtest_frame = pd.concat(
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
        backtest_frame.to_csv(backtest_dir / f"{safe_name}_best_mapping_backtest.csv")

    save_backtest_visualizations(
        output_dir=output_path,
        evaluation_results=evaluation_results,
        validation_ranking=validation_ranking,
        test_ranking=test_ranking,
    )


def _normalized_nav_from_returns(portfolio_returns: pd.Series, initial_capital: float) -> pd.Series:
    returns = portfolio_returns.dropna()
    if returns.empty:
        return pd.Series(dtype=float, name="normalized_nav")
    nav = (1.0 + returns).cumprod() * initial_capital
    return nav.rename("normalized_nav")


def _build_nav_frame(
    evaluation_results: list[dict],
    split_name: str | None = None,
    normalize_to_one: bool = True,
) -> pd.DataFrame:
    nav_series = {}
    for result in evaluation_results:
        experiment_name = result["summary"]["experiment_name"]
        portfolio_returns = result["backtest_result"]["portfolio_returns"].copy()
        if split_name is not None:
            split_dates = result["splits"][split_name]
            portfolio_returns = portfolio_returns.loc[portfolio_returns.index.intersection(split_dates)]
        nav = _normalized_nav_from_returns(portfolio_returns, initial_capital=1.0 if normalize_to_one else 100000.0)
        nav_series[experiment_name] = nav
    nav_frame = pd.concat(nav_series, axis=1).sort_index()
    if normalize_to_one and not nav_frame.empty:
        nav_frame = nav_frame.ffill()
    return nav_frame


def _add_split_boundaries(ax, split_summary: pd.DataFrame, y_min: float, y_max: float) -> None:
    color_map = {
        "train": "#d9edf7",
        "validation": "#fcf8e3",
        "test": "#dff0d8",
    }
    for _, row in split_summary.iterrows():
        if pd.isna(row["start_date"]) or pd.isna(row["end_date"]):
            continue
        if row.get("selection") != "ordered_oldest":
            continue
        ax.axvspan(row["start_date"], row["end_date"], alpha=0.12, color=color_map.get(row["split"], "#eeeeee"))

    for _, row in split_summary.iterrows():
        if pd.isna(row["start_date"]):
            continue
        if row.get("selection") != "ordered_oldest":
            continue
        if row["split"] == "train":
            continue
        ax.axvline(row["start_date"], color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(
            row["start_date"],
            y_max,
            f" {row['split']}",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color="gray",
        )


def save_nav_comparison_png(
    nav_frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
    ylabel: str,
    ranking_df: pd.DataFrame | None = None,
    split_summary: pd.DataFrame | None = None,
) -> None:
    if nav_frame.empty:
        return

    ordered_columns = list(nav_frame.columns)
    if ranking_df is not None and not ranking_df.empty and "experiment_name" in ranking_df.columns:
        ranked = [name for name in ranking_df["experiment_name"] if name in nav_frame.columns]
        ordered_columns = ranked + [name for name in nav_frame.columns if name not in ranked]

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_frame = nav_frame.loc[:, ordered_columns]
    for column in plot_frame.columns:
        ax.plot(plot_frame.index, plot_frame[column], linewidth=1.8, label=column)

    if split_summary is not None and not split_summary.empty:
        valid_values = plot_frame.to_numpy(dtype=float)
        y_min = float(np.nanmin(valid_values))
        y_max = float(np.nanmax(valid_values))
        _add_split_boundaries(ax, split_summary, y_min=y_min, y_max=y_max)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_nav_comparison_gif(
    nav_frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
    ylabel: str,
    ranking_df: pd.DataFrame | None = None,
    fps: int = 12,
) -> bool:
    if nav_frame.empty:
        return False

    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception:
        return False

    ordered_columns = list(nav_frame.columns)
    if ranking_df is not None and not ranking_df.empty and "experiment_name" in ranking_df.columns:
        ranked = [name for name in ranking_df["experiment_name"] if name in nav_frame.columns]
        ordered_columns = ranked + [name for name in nav_frame.columns if name not in ranked]

    plot_frame = nav_frame.loc[:, ordered_columns]
    dates = plot_frame.index
    if len(dates) < 2:
        return False

    fig, (ax, ax_rank) = plt.subplots(
        ncols=2,
        figsize=(18, 8),
        gridspec_kw={"width_ratios": [4.8, 1.8]},
    )
    lines = {column: ax.plot([], [], linewidth=2.0, label=column)[0] for column in plot_frame.columns}
    ax.set_xlim(dates.min(), dates.max())
    valid_values = plot_frame.to_numpy(dtype=float)
    y_min = float(np.nanmin(valid_values))
    y_max = float(np.nanmax(valid_values))
    pad = max((y_max - y_min) * 0.08, 1e-6)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)

    ax_rank.set_axis_off()
    ax_rank.set_xlim(0.0, 1.0)
    ax_rank.set_ylim(0.0, 1.0)

    frame_indices = np.linspace(1, len(plot_frame), num=min(len(plot_frame), 80), dtype=int)

    def update(frame_end: int):
        current = plot_frame.iloc[:frame_end]
        for column, line in lines.items():
            line.set_data(current.index, current[column].to_numpy())

        latest_values = current.iloc[-1].sort_values(ascending=False)
        ax_rank.clear()
        ax_rank.set_axis_off()
        ax_rank.set_xlim(0.0, 1.0)
        ax_rank.set_ylim(0.0, 1.0)
        ax_rank.set_title("Current Ranking", fontsize=12, loc="left", pad=10)

        n_items = len(latest_values)
        top_margin = 0.94
        bottom_margin = 0.04
        step = (top_margin - bottom_margin) / max(n_items, 1)
        font_size = max(7.5, min(10.5, 12.5 - 0.15 * max(0, n_items - 8)))

        for rank, (column, latest_value) in enumerate(latest_values.items(), start=1):
            y = top_margin - (rank - 1) * step
            line_color = lines[column].get_color()
            label_text = f"{rank:>2}. {column[:32]}"
            value_text = f"{latest_value:.3f}"
            ax_rank.text(
                0.02,
                y,
                label_text,
                fontsize=font_size,
                color=line_color,
                ha="left",
                va="center",
                family="monospace",
                fontweight="bold" if rank <= 3 else "normal",
            )
            ax_rank.text(
                0.98,
                y,
                value_text,
                fontsize=font_size,
                color=line_color,
                ha="right",
                va="center",
                family="monospace",
            )

        ax.set_title(f"{title}\nThrough {current.index[-1].date()}", fontsize=14)
        return list(lines.values())

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=max(40, int(1000 / fps)),
        blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return True


def save_backtest_visualizations(
    output_dir: str | Path,
    evaluation_results: list[dict],
    validation_ranking: pd.DataFrame,
    test_ranking: pd.DataFrame,
) -> None:
    output_path = Path(output_dir)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    split_summary = evaluation_results[0]["split_summary"] if evaluation_results else pd.DataFrame()

    full_nav = _build_nav_frame(evaluation_results, split_name=None, normalize_to_one=True)
    validation_nav = _build_nav_frame(evaluation_results, split_name="validation", normalize_to_one=True)
    test_nav = _build_nav_frame(evaluation_results, split_name="test", normalize_to_one=True)

    save_nav_comparison_png(
        nav_frame=full_nav,
        output_path=plots_dir / "full_period_nav_comparison.png",
        title="Backtest NAV Comparison Across Models",
        ylabel="Normalized NAV",
        ranking_df=test_ranking,
        split_summary=split_summary,
    )
    save_nav_comparison_png(
        nav_frame=validation_nav,
        output_path=plots_dir / "validation_nav_comparison.png",
        title="Validation NAV Comparison",
        ylabel="Normalized NAV",
        ranking_df=validation_ranking,
    )
    save_nav_comparison_png(
        nav_frame=test_nav,
        output_path=plots_dir / "test_nav_comparison.png",
        title="Test NAV Comparison",
        ylabel="Normalized NAV",
        ranking_df=test_ranking,
    )

    gif_saved = save_nav_comparison_gif(
        nav_frame=test_nav,
        output_path=plots_dir / "test_nav_comparison.gif",
        title="Test NAV Comparison Animation",
        ylabel="Normalized NAV",
        ranking_df=test_ranking,
    )
    status_frame = pd.DataFrame(
        [
            {"artifact": "full_period_nav_comparison.png", "saved": True},
            {"artifact": "validation_nav_comparison.png", "saved": True},
            {"artifact": "test_nav_comparison.png", "saved": True},
            {"artifact": "test_nav_comparison.gif", "saved": bool(gif_saved)},
        ]
    )
    status_frame.to_csv(plots_dir / "plot_manifest.csv", index=False)


__all__ = [
    "ActionBacktestConfig",
    "build_default_action_library",
    "load_tradable_returns",
    "build_hmm_baseline_experiment_result",
    "split_window_dates",
    "summarize_split_dates",
    "enumerate_cluster_action_mappings",
    "serialize_mapping",
    "build_weight_frame_from_mapping",
    "run_action_mapping_backtest",
    "calculate_backtest_metrics",
    "search_best_cluster_action_mapping",
    "evaluate_model_cluster_actions",
    "evaluate_random_choice_baseline",
    "build_model_ranking_tables",
    "save_evaluation_artifacts",
]
