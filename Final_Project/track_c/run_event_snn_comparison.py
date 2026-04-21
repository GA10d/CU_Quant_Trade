from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


TRACK_C_DIR = Path(__file__).resolve().parent
TRACK_B_DIR = TRACK_C_DIR.parent / "track_b"
if str(TRACK_B_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_B_DIR))


from cluster_action_backtest import (  # noqa: E402
    _metric_sort_key,
    ActionBacktestConfig,
    build_default_action_library,
    build_model_ranking_tables,
    evaluate_model_cluster_actions,
    load_tradable_returns,
    save_evaluation_artifacts,
)
from encoder_only_transformer import ClusteringConfig, HMMReferenceConfig  # noqa: E402
from rl_backtest_agent import (  # noqa: E402
    QLearningAgentConfig,
    evaluate_model_with_q_learning,
    save_rl_ranking_artifacts,
)

from event_snn_pipeline import (  # noqa: E402
    EventEncodingConfig,
    EventSelectionConfig,
    build_default_event_model_config,
    build_default_event_training_config,
    build_default_track_c_data_config,
    run_event_snn_experiment,
    save_event_snn_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Track C event-encoded SNN benchmark and compare it with the existing Track B leaderboard.",
    )
    parser.add_argument("--experiment-name", type=str, default="snn_event_lif")
    parser.add_argument("--cluster-count", type=int, default=4, choices=[3, 4])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--objective", type=str, default="sharpe")
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    parser.add_argument("--execution-lag", type=int, default=1)
    parser.add_argument("--assets", nargs="+", default=["SPY", "TLT", "GLD"])
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.985)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quantile-lookback", type=int, default=63)
    parser.add_argument("--breakout-lookback", type=int, default=20)
    parser.add_argument("--volatility-lookback", type=int, default=20)
    parser.add_argument("--volatility-quantile-lookback", type=int, default=126)
    parser.add_argument("--volume-zscore-lookback", type=int, default=63)
    parser.add_argument("--upper-quantile", type=float, default=0.80)
    parser.add_argument("--lower-quantile", type=float, default=0.20)
    parser.add_argument("--volume-spike-z", type=float, default=2.0)
    parser.add_argument("--volume-dryup-z", type=float, default=-1.0)
    parser.add_argument("--disable-cross-asset-events", action="store_true", default=False)
    parser.add_argument("--prune-low-value-events", action="store_true", default=False)
    parser.add_argument("--min-activation-rate", type=float, default=0.0)
    parser.add_argument("--max-activation-rate", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--min-mutual-info", type=float, default=None)
    parser.add_argument("--allow-prune-cross-asset-events", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _prefix_rankings(frame: pd.DataFrame, source_family: str) -> pd.DataFrame:
    prefixed = frame.copy()
    prefixed["source_family"] = source_family
    prefixed["raw_experiment_name"] = prefixed["experiment_name"]
    prefix_map = {
        "cluster_action": "CA",
        "rl": "RL",
    }
    prefix = prefix_map.get(source_family, source_family.upper())
    prefixed["experiment_name"] = prefix + ":" + prefixed["experiment_name"].astype(str)
    return prefixed


def _build_overall_leaderboard(
    validation_ranking: pd.DataFrame,
    test_ranking: pd.DataFrame,
) -> pd.DataFrame:
    validation_columns = [
        "experiment_name",
        "source_family",
        "raw_experiment_name",
        "architecture",
        "validation_rank",
        "validation_sharpe",
        "validation_annual_return",
        "validation_max_drawdown",
    ]
    test_columns = [
        "experiment_name",
        "test_rank",
        "test_sharpe",
        "test_annual_return",
        "test_max_drawdown",
    ]
    leaderboard = validation_ranking.loc[:, validation_columns].merge(
        test_ranking.loc[:, test_columns],
        on="experiment_name",
        how="left",
    )
    leaderboard["average_rank"] = (
        leaderboard["validation_rank"].astype(float) + leaderboard["test_rank"].astype(float)
    ) / 2.0
    leaderboard["rank_gap"] = leaderboard["test_rank"].astype(float) - leaderboard["validation_rank"].astype(float)
    leaderboard = leaderboard.sort_values(
        ["average_rank", "test_rank", "validation_rank", "experiment_name"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    leaderboard.insert(0, "overall_rank", np.arange(1, len(leaderboard) + 1))
    return leaderboard


def main() -> None:
    args = parse_args()
    track_b_results_dir = TRACK_B_DIR / "results" / f"combined_nav_rankings_k{args.cluster_count}"

    output_root = Path(args.output_dir) if args.output_dir else TRACK_C_DIR / "results"
    experiment_root = output_root / args.experiment_name
    model_output_dir = experiment_root / "models"
    ca_output_dir = experiment_root / f"cluster_action_rankings_k{args.cluster_count}"
    rl_output_dir = experiment_root / f"rl_backtest_rankings_k{args.cluster_count}"
    comparison_output_dir = experiment_root / f"comparison_vs_track_b_k{args.cluster_count}"
    comparison_output_dir.mkdir(parents=True, exist_ok=True)

    data_config = build_default_track_c_data_config(TRACK_C_DIR.parent / "data")
    model_config = build_default_event_model_config()
    event_config = EventEncodingConfig(
        quantile_lookback=args.quantile_lookback,
        breakout_lookback=args.breakout_lookback,
        volatility_lookback=args.volatility_lookback,
        volatility_quantile_lookback=args.volatility_quantile_lookback,
        volume_zscore_lookback=args.volume_zscore_lookback,
        upper_quantile=args.upper_quantile,
        lower_quantile=args.lower_quantile,
        volume_spike_z=args.volume_spike_z,
        volume_dryup_z=args.volume_dryup_z,
        include_cross_asset_events=not args.disable_cross_asset_events,
    )
    selection_config = EventSelectionConfig(
        enabled=args.prune_low_value_events,
        min_activation_rate=args.min_activation_rate,
        max_activation_rate=args.max_activation_rate,
        max_features=args.max_features,
        min_mutual_info=args.min_mutual_info,
        preserve_cross_asset_events=not args.allow_prune_cross_asset_events,
    )
    training_config = replace(
        build_default_event_training_config(device=args.device),
        train_ratio=1.0 - args.validation_ratio - args.test_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        device=args.device,
    )
    clustering_config = ClusteringConfig(
        target_cluster_count=args.cluster_count,
        cluster_candidates=(args.cluster_count,),
        n_init=20,
        batch_size=256,
    )
    hmm_config = HMMReferenceConfig(
        enabled=False,
        state_candidates=(args.cluster_count,),
        covariance_type="diag",
        n_restarts=6,
        n_iter=500,
    )

    experiment_result = run_event_snn_experiment(
        experiment_name=args.experiment_name,
        data_config=data_config,
        event_config=event_config,
        selection_config=selection_config,
        model_config=model_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )
    save_path = save_event_snn_experiment(experiment_result, model_output_dir)
    print(f"Saved Track C experiment artifacts to: {save_path}")
    print(
        "Event panel:",
        f"features={experiment_result['summary']['input_dim']},",
        f"density={experiment_result['sequence_metadata'].get('event_density', float('nan')):.4f}",
    )
    if experiment_result["sequence_metadata"].get("selection_applied"):
        print(
            "Selected panel:",
            f"selected_density={experiment_result['sequence_metadata'].get('selected_event_density', float('nan')):.4f},",
            f"selection_applied={experiment_result['sequence_metadata'].get('selection_applied')}",
        )

    backtest_config = ActionBacktestConfig(
        tradable_assets=tuple(args.assets),
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        risk_free_rate=args.risk_free_rate,
        execution_lag=args.execution_lag,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        objective=args.objective,
        allow_action_reuse=False,
        split_random_state=args.random_state,
    )
    action_library = build_default_action_library(
        n_actions=args.cluster_count,
        tradable_assets=backtest_config.tradable_assets,
    )
    asset_returns = load_tradable_returns(
        data_config=data_config,
        tradable_assets=backtest_config.tradable_assets,
    )

    ca_result = evaluate_model_cluster_actions(
        experiment_result=experiment_result,
        asset_returns=asset_returns,
        action_library=action_library,
        config=backtest_config,
    )
    ca_validation, ca_test, ca_combined = build_model_ranking_tables(
        evaluation_results=[ca_result],
        objective=args.objective,
    )
    save_evaluation_artifacts(
        output_dir=ca_output_dir,
        evaluation_results=[ca_result],
        validation_ranking=ca_validation,
        test_ranking=ca_test,
        combined_ranking=ca_combined,
    )

    agent_config = QLearningAgentConfig(
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        include_previous_action=True,
        eval_interval=args.eval_interval,
        random_state=args.random_state,
        reward_scale=args.reward_scale,
    )
    rl_result = evaluate_model_with_q_learning(
        experiment_result=experiment_result,
        asset_returns=asset_returns,
        action_library=action_library,
        backtest_config=backtest_config,
        agent_config=agent_config,
    )
    rl_validation, rl_test, rl_combined = build_model_ranking_tables(
        evaluation_results=[rl_result],
        objective=args.objective,
    )
    save_rl_ranking_artifacts(
        output_dir=rl_output_dir,
        evaluation_results=[rl_result],
        validation_ranking=rl_validation,
        test_ranking=rl_test,
        combined_ranking=rl_combined,
    )

    track_b_validation = pd.read_csv(track_b_results_dir / "combined_validation_ranking.csv")
    track_b_test = pd.read_csv(track_b_results_dir / "combined_test_ranking.csv")

    augmented_validation = pd.concat(
        [
            track_b_validation,
            _prefix_rankings(ca_validation, "cluster_action"),
            _prefix_rankings(rl_validation, "rl"),
        ],
        ignore_index=True,
    )
    augmented_test = pd.concat(
        [
            track_b_test,
            _prefix_rankings(ca_test, "cluster_action"),
            _prefix_rankings(rl_test, "rl"),
        ],
        ignore_index=True,
    )

    objective_column_validation = f"validation_{args.objective}"
    objective_column_test = f"test_{args.objective}"
    _, descending = _metric_sort_key(args.objective)

    augmented_validation = augmented_validation.drop(columns=["validation_rank"], errors="ignore")
    augmented_validation = augmented_validation.sort_values(
        objective_column_validation,
        ascending=not descending,
    ).reset_index(drop=True)
    augmented_validation.insert(0, "validation_rank", range(1, len(augmented_validation) + 1))

    augmented_test = augmented_test.drop(columns=["test_rank"], errors="ignore")
    augmented_test = augmented_test.sort_values(
        objective_column_test,
        ascending=not descending,
    ).reset_index(drop=True)
    augmented_test.insert(0, "test_rank", range(1, len(augmented_test) + 1))

    overall_leaderboard = _build_overall_leaderboard(
        validation_ranking=augmented_validation,
        test_ranking=augmented_test,
    )

    augmented_validation.to_csv(comparison_output_dir / "combined_validation_ranking_with_track_c.csv", index=False)
    augmented_test.to_csv(comparison_output_dir / "combined_test_ranking_with_track_c.csv", index=False)
    overall_leaderboard.to_csv(comparison_output_dir / "overall_model_leaderboard_with_track_c.csv", index=False)

    track_c_rows = overall_leaderboard.loc[
        overall_leaderboard["raw_experiment_name"].astype(str).str.startswith(args.experiment_name),
        [
            "overall_rank",
            "experiment_name",
            "source_family",
            "validation_rank",
            "test_rank",
            "average_rank",
            "validation_sharpe",
            "test_sharpe",
        ],
    ]
    track_c_rows.to_csv(comparison_output_dir / "track_c_rows_only.csv", index=False)

    print()
    print("Track C rows inside the augmented overall leaderboard:")
    if track_c_rows.empty:
        print("No Track C rows were found in the merged leaderboard.")
    else:
        print(track_c_rows.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(f"Saved Track C comparison artifacts to: {comparison_output_dir}")


if __name__ == "__main__":
    main()
