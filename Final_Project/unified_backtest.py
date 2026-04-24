from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
TRACK_B_DIR = ROOT_DIR / "track_b"
TRACK_A_DIR = ROOT_DIR / "track_a"
for p in (ROOT_DIR, TRACK_B_DIR, TRACK_A_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---------------------------------------------------------------------
# Reuse Track B utilities so Track A / Track B share the same backtest
# engine, mapping search, metrics, and output logic.
# ---------------------------------------------------------------------
from track_b_backtest import (  # type: ignore
    DEFAULT_MODELS_DIR,
    TrackBBacktestConfig,
    build_default_action_library,
    build_nav_frame,
    build_ranking_tables,
    calculate_metrics,
    evaluate_single_experiment,
    load_tradable_returns,
    metric_sort_direction,
    resolve_fallback_action_name,
    run_backtest,
    save_nav_plot,
    save_single_experiment_outputs,
    search_best_mapping,
    split_window_dates,
    summarize_split_dates,
)
from experiment_cache import run_experiment_with_cache  # type: ignore
from experiment_presets import (  # type: ignore
    build_architecture_runners,
    build_default_experiment_setups,
)
from encoder_only_transformer import split_ordered_train_random_holdout_indices  # type: ignore


# Optional Track A imports.
try:
    from hmm_regime_detection import HMMRegimeDetector  # type: ignore
except Exception:
    HMMRegimeDetector = None


@dataclass
class UnifiedBacktestConfig:
    tracks: tuple[str, ...] = ("a", "b")
    track_b_architectures: tuple[str, ...] | None = None
    cluster_count: int = 4
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
    device: str = "auto"
    models_dir: str = str(DEFAULT_MODELS_DIR)
    output_dir: str | None = None

    # Track A inputs
    track_a_features_path: str | None = None
    track_a_regimes_path: str | None = None
    track_a_market_data_path: str | None = None
    track_a_states: int = 4
    track_a_hmm_covariance_type: str = "diag"
    track_a_hmm_n_iter: int = 1000
    track_a_hmm_restarts: int = 10

    @property
    def as_track_b_config(self) -> TrackBBacktestConfig:
        return TrackBBacktestConfig(
            tradable_assets=self.tradable_assets,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            risk_free_rate=self.risk_free_rate,
            annualization_factor=self.annualization_factor,
            execution_lag=self.execution_lag,
            validation_ratio=self.validation_ratio,
            test_ratio=self.test_ratio,
            objective=self.objective,
            allow_action_reuse=self.allow_action_reuse,
            fallback_action_name=self.fallback_action_name,
            split_random_state=self.split_random_state,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Track A / Track B backtest with shared evaluation logic."
    )
    parser.add_argument("--tracks", nargs="+", default=["a", "b"], choices=["a", "b"])
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
    parser.add_argument("--track-b-architectures", nargs="*", default=None)
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--track-a-features-path", type=str, default=None)
    parser.add_argument("--track-a-regimes-path", type=str, default=None)
    parser.add_argument("--track-a-market-data-path", type=str, default=None)
    parser.add_argument("--track-a-states", type=int, default=4)
    parser.add_argument("--track-a-hmm-covariance-type", type=str, default="diag")
    parser.add_argument("--track-a-hmm-n-iter", type=int, default=1000)
    parser.add_argument("--track-a-hmm-restarts", type=int, default=10)
    return parser.parse_args()


def normalize_regime_series(regime_series: pd.Series) -> pd.Series:
    s = regime_series.copy().dropna()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    if s.empty:
        raise ValueError("Track A regime series is empty after cleaning.")

    # Normalize labels to integer ids for fair mapping search.
    categories = pd.Categorical(s.astype(str))
    return pd.Series(categories.codes, index=s.index, name="cluster")


def infer_track_a_regimes(config: UnifiedBacktestConfig) -> pd.Series:
    if config.track_a_regimes_path:
        regime_path = Path(config.track_a_regimes_path)
        if not regime_path.exists():
            raise FileNotFoundError(f"Track A regimes file not found: {regime_path}")

        df = pd.read_csv(regime_path, index_col=0, parse_dates=True)
        if "regime" in df.columns:
            return normalize_regime_series(df["regime"])
        if df.shape[1] == 1:
            return normalize_regime_series(df.iloc[:, 0])
        raise ValueError(
            "Track A regimes CSV must contain a 'regime' column or exactly one column."
        )

    if config.track_a_features_path is None:
        raise ValueError(
            "Track A requires either --track-a-regimes-path or --track-a-features-path."
        )
    if HMMRegimeDetector is None:
        raise ImportError(
            "Could not import HMMRegimeDetector. Make sure hmm_regime_detection.py is available."
        )

    feature_path = Path(config.track_a_features_path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Track A features file not found: {feature_path}")

    features = pd.read_csv(feature_path, index_col=0, parse_dates=True).sort_index()
    detector = HMMRegimeDetector(
        n_states=config.track_a_states,
        covariance_type=config.track_a_hmm_covariance_type,
        n_iter=config.track_a_hmm_n_iter,
        random_state=config.split_random_state,
    )
    detector.fit(features, n_restarts=config.track_a_hmm_restarts)
    if hasattr(detector, "label_regimes"):
        try:
            detector.label_regimes(features)
        except Exception:
            pass
    raw_states = pd.Series(detector.states, index=features.index, name="regime")
    return normalize_regime_series(raw_states)


def split_series_dates(
    dates: pd.Index,
    validation_ratio: float,
    test_ratio: float,
    random_state: int,
) -> dict[str, pd.Index]:
    n_obs = len(dates)
    if n_obs < 10:
        raise ValueError(f"Need at least 10 observations, got {n_obs}")

    train_ratio = 1.0 - validation_ratio - test_ratio
    split_indices = split_ordered_train_random_holdout_indices(
        n_obs=n_obs,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    return {
        "train": pd.Index(dates[split_indices["train"]], name="Date"),
        "validation": pd.Index(dates[split_indices["validation"]], name="Date"),
        "test": pd.Index(dates[split_indices["test"]], name="Date"),
    }


def evaluate_track_a(
    regime_series: pd.Series,
    asset_returns: pd.DataFrame,
    action_library: dict[str, pd.Series],
    config: TrackBBacktestConfig,
) -> dict:
    common_dates = regime_series.index.intersection(asset_returns.index)
    common_dates = common_dates.sort_values()
    if len(common_dates) < 10:
        raise ValueError("Track A has too few overlapping dates with asset returns.")

    regime_series = regime_series.loc[common_dates]
    splits = split_series_dates(
        dates=common_dates,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        random_state=config.split_random_state,
    )
    split_summary = summarize_split_dates(splits)

    search_result = search_best_mapping(
        cluster_series=regime_series,
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
        "experiment_name": "track_a_hmm",
        "track": "A",
        "architecture": "hmm",
        "n_clusters": int(np.unique(regime_series.dropna()).size),
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


def build_static_weight_frame(
    dates: pd.Index,
    assets: tuple[str, ...],
    weights: dict[str, float],
) -> pd.DataFrame:
    normalized = pd.Series(weights, dtype=float).reindex(list(assets)).fillna(0.0)
    total = float(normalized.sum())
    if total <= 0:
        raise ValueError("Static baseline weights must sum to a positive value.")
    normalized = normalized / total
    return pd.DataFrame([normalized.values] * len(dates), index=dates, columns=list(assets))


def evaluate_static_baseline(
    name: str,
    weights: dict[str, float],
    reference_splits: dict[str, pd.Index],
    asset_returns: pd.DataFrame,
    config: TrackBBacktestConfig,
    track_label: str,
) -> dict:
    common_dates = asset_returns.index.intersection(
        reference_splits["train"].append(reference_splits["validation"]).append(reference_splits["test"])
    )
    common_dates = common_dates.sort_values()
    weight_frame = build_static_weight_frame(common_dates, config.tradable_assets, weights)

    pseudo_cluster = pd.Series(0, index=common_dates, name="cluster")
    pseudo_action_library = {name: pd.Series(weights, dtype=float).reindex(config.tradable_assets).fillna(0.0)}
    pseudo_mapping = {0: name}
    bt = run_backtest(
        cluster_series=pseudo_cluster,
        asset_returns=asset_returns,
        action_library=pseudo_action_library,
        mapping=pseudo_mapping,
        config=config,
    )
    bt["target_weights"] = weight_frame.reindex(bt["target_weights"].index).ffill().fillna(0.0)
    bt["executed_weights"] = weight_frame.reindex(bt["executed_weights"].index).ffill().fillna(0.0)
    bt["action_series"] = pd.Series(name, index=bt["portfolio_returns"].index, name="action")

    validation_returns = bt["portfolio_returns"].loc[
        bt["portfolio_returns"].index.intersection(reference_splits["validation"])
    ]
    validation_turnover = bt["turnover"].loc[validation_returns.index]
    validation_metrics = calculate_metrics(validation_returns, validation_turnover, config)

    test_returns = bt["portfolio_returns"].loc[
        bt["portfolio_returns"].index.intersection(reference_splits["test"])
    ]
    test_turnover = bt["turnover"].loc[test_returns.index]
    test_metrics = calculate_metrics(test_returns, test_turnover, config)

    full_metrics = calculate_metrics(bt["portfolio_returns"], bt["turnover"], config)

    summary = {
        "experiment_name": name,
        "track": track_label,
        "architecture": "baseline",
        "n_clusters": 1,
        "best_mapping": name,
        "objective": config.objective,
    }
    summary.update({f"validation_{k}": v for k, v in validation_metrics.items()})
    summary.update({f"test_{k}": v for k, v in test_metrics.items()})
    summary.update({f"full_{k}": v for k, v in full_metrics.items()})

    return {
        "summary": summary,
        "splits": reference_splits,
        "split_summary": summarize_split_dates(reference_splits),
        "best_mapping": {0: name},
        "best_mapping_text": name,
        "search_results": pd.DataFrame([{ "mapping": name, "objective_value": np.nan }]),
        "backtest_result": bt,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "full_metrics": full_metrics,
    }


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

    for result in evaluation_results:
        save_single_experiment_outputs(output_dir, result)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    full_nav = build_nav_frame(evaluation_results, split_name=None)
    validation_nav = build_nav_frame(evaluation_results, split_name="validation")
    test_nav = build_nav_frame(evaluation_results, split_name="test")

    full_nav.to_csv(plots_dir / "full_period_nav.csv")
    validation_nav.to_csv(plots_dir / "validation_nav.csv")
    test_nav.to_csv(plots_dir / "test_nav.csv")

    save_nav_plot(full_nav, plots_dir / "full_period_nav.png", "Unified Full-Period NAV Comparison")
    save_nav_plot(validation_nav, plots_dir / "validation_nav.png", "Unified Validation NAV Comparison")
    save_nav_plot(test_nav, plots_dir / "test_nav.png", "Unified Test NAV Comparison")


def run_track_b(
    unified_config: UnifiedBacktestConfig,
    backtest_config: TrackBBacktestConfig,
    action_library: dict[str, pd.Series],
) -> list[dict]:
    runners = build_architecture_runners()
    data_dir = ROOT_DIR / "data"

    experiment_setups = build_default_experiment_setups(
        data_dir=data_dir,
        hmm_enabled=False,
        target_cluster_count=unified_config.cluster_count,
        device=unified_config.device,
    )

    if unified_config.track_b_architectures:
        requested = set(unified_config.track_b_architectures)
        experiment_setups = [setup for setup in experiment_setups if setup["architecture"] in requested]

    if not experiment_setups:
        raise ValueError("No Track B experiment setups selected. Check --track-b-architectures.")

    experiment_setups = [
        {
            **setup,
            "training_config": replace(
                setup["training_config"],
                train_ratio=backtest_config.train_ratio,
                validation_ratio=unified_config.validation_ratio,
                test_ratio=unified_config.test_ratio,
                device=unified_config.device,
            ),
        }
        for setup in experiment_setups
    ]

    returns_data_config = experiment_setups[0]["data_config"]
    asset_returns = load_tradable_returns(
        data_config=returns_data_config,
        tradable_assets=backtest_config.tradable_assets,
    )

    evaluation_results: list[dict] = []
    for setup in experiment_setups:
        print(f"=== Running Track B: {setup['name']} ===")
        runner = runners[setup["architecture"]]
        experiment_result = run_experiment_with_cache(
            runner=runner,
            experiment_name=setup["name"],
            data_config=setup["data_config"],
            model_config=setup["model_config"],
            training_config=setup["training_config"],
            clustering_config=setup["clustering_config"],
            hmm_config=setup["hmm_config"],
            models_dir=unified_config.models_dir,
            verbose=True,
        )
        result = evaluate_single_experiment(
            experiment_result=experiment_result,
            asset_returns=asset_returns,
            action_library=action_library,
            config=backtest_config,
        )
        result["summary"]["track"] = "B"
        evaluation_results.append(result)

    # Add shared baselines on Track B date split using the first result.
    if evaluation_results:
        reference_splits = evaluation_results[0]["splits"]
        evaluation_results.append(
            evaluate_static_baseline(
                name="baseline_equal_weight_B",
                weights={asset: 1 / len(backtest_config.tradable_assets) for asset in backtest_config.tradable_assets},
                reference_splits=reference_splits,
                asset_returns=asset_returns,
                config=backtest_config,
                track_label="B",
            )
        )
        sixty_forty = {backtest_config.tradable_assets[0]: 0.6, backtest_config.tradable_assets[1]: 0.4}
        evaluation_results.append(
            evaluate_static_baseline(
                name="baseline_60_40_B",
                weights=sixty_forty,
                reference_splits=reference_splits,
                asset_returns=asset_returns,
                config=backtest_config,
                track_label="B",
            )
        )
    return evaluation_results


def run_track_a(
    unified_config: UnifiedBacktestConfig,
    backtest_config: TrackBBacktestConfig,
    action_library: dict[str, pd.Series],
) -> list[dict]:
    if unified_config.track_a_market_data_path:
        market_data_path = Path(unified_config.track_a_market_data_path)
        if not market_data_path.exists():
            raise FileNotFoundError(f"Track A market data file not found: {market_data_path}")
        market_data = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
        missing_assets = [a for a in backtest_config.tradable_assets if a not in market_data.columns]
        if missing_assets:
            raise ValueError(f"Track A market data missing assets: {missing_assets}")
        asset_returns = market_data.loc[:, list(backtest_config.tradable_assets)].pct_change(fill_method=None).dropna(how="all")
        asset_returns.index.name = "Date"
    else:
        # Fall back to the same data source used by Track B for fairness if no separate Track A market file is given.
        experiment_setups = build_default_experiment_setups(
            data_dir=ROOT_DIR / "data",
            hmm_enabled=False,
            target_cluster_count=unified_config.cluster_count,
            device=unified_config.device,
        )
        asset_returns = load_tradable_returns(
            data_config=experiment_setups[0]["data_config"],
            tradable_assets=backtest_config.tradable_assets,
        )

    regime_series = infer_track_a_regimes(unified_config)
    result = evaluate_track_a(
        regime_series=regime_series,
        asset_returns=asset_returns,
        action_library=action_library,
        config=backtest_config,
    )
    results = [result]

    reference_splits = result["splits"]
    results.append(
        evaluate_static_baseline(
            name="baseline_equal_weight_A",
            weights={asset: 1 / len(backtest_config.tradable_assets) for asset in backtest_config.tradable_assets},
            reference_splits=reference_splits,
            asset_returns=asset_returns,
            config=backtest_config,
            track_label="A",
        )
    )
    sixty_forty = {backtest_config.tradable_assets[0]: 0.6, backtest_config.tradable_assets[1]: 0.4}
    results.append(
        evaluate_static_baseline(
            name="baseline_60_40_A",
            weights=sixty_forty,
            reference_splits=reference_splits,
            asset_returns=asset_returns,
            config=backtest_config,
            track_label="A",
        )
    )
    return results


def print_rankings(validation_ranking: pd.DataFrame, test_ranking: pd.DataFrame, overall_ranking: pd.DataFrame, objective: str) -> None:
    validation_cols = [
        "validation_rank",
        "track",
        "experiment_name",
        "architecture",
        f"validation_{objective}",
        "validation_annual_return",
        "validation_max_drawdown",
        "best_mapping",
    ]
    test_cols = [
        "test_rank",
        "track",
        "experiment_name",
        "architecture",
        f"test_{objective}",
        "test_annual_return",
        "test_max_drawdown",
        "best_mapping",
    ]
    overall_cols = [
        "overall_rank",
        "track",
        "experiment_name",
        "architecture",
        "validation_rank",
        "test_rank",
        "avg_rank",
        f"validation_{objective}",
        f"test_{objective}",
    ]

    print("=" * 80)
    print("Validation Ranking")
    print("=" * 80)
    print(validation_ranking[validation_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    print("=" * 80)
    print("Test Ranking")
    print("=" * 80)
    print(test_ranking[test_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    print("=" * 80)
    print("Overall Ranking")
    print("=" * 80)
    print(overall_ranking[overall_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()


def main() -> None:
    args = parse_args()
    unified_config = UnifiedBacktestConfig(
        tracks=tuple(args.tracks),
        track_b_architectures=tuple(args.track_b_architectures) if args.track_b_architectures else None,
        cluster_count=args.cluster_count,
        tradable_assets=tuple(args.assets),
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        risk_free_rate=args.risk_free_rate,
        execution_lag=args.execution_lag,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        objective=args.objective,
        allow_action_reuse=args.allow_action_reuse,
        device=args.device,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        track_a_features_path=args.track_a_features_path,
        track_a_regimes_path=args.track_a_regimes_path,
        track_a_market_data_path=args.track_a_market_data_path,
        track_a_states=args.track_a_states,
        track_a_hmm_covariance_type=args.track_a_hmm_covariance_type,
        track_a_hmm_n_iter=args.track_a_hmm_n_iter,
        track_a_hmm_restarts=args.track_a_hmm_restarts,
    )
    backtest_config = unified_config.as_track_b_config
    action_library = build_default_action_library(
        n_actions=unified_config.cluster_count,
        tradable_assets=backtest_config.tradable_assets,
    )

    output_dir = Path(
        unified_config.output_dir
        if unified_config.output_dir
        else ROOT_DIR / "artifacts" / f"unified_backtest_k{unified_config.cluster_count}"
    )

    print("=" * 80)
    print("Unified Backtest")
    print("=" * 80)
    print(f"Tracks: {', '.join(unified_config.tracks)}")
    print(f"Objective: {unified_config.objective}")
    print("Action library:")
    print(pd.DataFrame(action_library).T.to_string(float_format=lambda x: f"{x:.2f}"))
    print()

    evaluation_results: list[dict] = []
    if "a" in unified_config.tracks:
        evaluation_results.extend(run_track_a(unified_config, backtest_config, action_library))
    if "b" in unified_config.tracks:
        evaluation_results.extend(run_track_b(unified_config, backtest_config, action_library))

    validation_ranking, test_ranking, overall_ranking = build_ranking_tables(
        evaluation_results=evaluation_results,
        objective=unified_config.objective,
    )
    save_all_outputs(
        output_dir=output_dir,
        evaluation_results=evaluation_results,
        validation_ranking=validation_ranking,
        test_ranking=test_ranking,
        overall_ranking=overall_ranking,
        action_library=action_library,
    )
    print_rankings(validation_ranking, test_ranking, overall_ranking, unified_config.objective)
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
