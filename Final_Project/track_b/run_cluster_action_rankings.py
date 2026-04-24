from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from cluster_action_backtest import (
    ActionBacktestConfig,
    build_hmm_baseline_experiment_result,
    build_default_action_library,
    build_model_ranking_tables,
    evaluate_model_cluster_actions,
    evaluate_random_choice_baseline,
    load_tradable_returns,
    save_evaluation_artifacts,
)
from encoder_only_transformer import HMMLEARN_AVAILABLE
from experiment_cache import DEFAULT_MODELS_DIR, run_experiment_with_cache
from experiment_presets import (
    build_architecture_runners,
    build_default_experiment_setups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cluster->action backtests for all Track B models and print validation/test rankings.",
    )
    parser.add_argument("--cluster-count", type=int, default=4, choices=[3, 4])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--objective", type=str, default="sharpe")
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--initial-capital", type=float, default=1.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    parser.add_argument("--cash-proxy-source", choices=["FLAT", "RF", "DGS3MO", "FEDFUNDS"], default="FLAT")
    parser.add_argument("--execution-lag", type=int, default=1)
    parser.add_argument("--mapping-fit-split", choices=["train", "validation"], default="train")
    parser.add_argument("--allow-action-reuse", action="store_true", default=False)
    parser.add_argument("--no-action-reuse", dest="allow_action_reuse", action="store_false")
    parser.add_argument("--assets", nargs="+", default=["SPY", "TLT", "GLD", "Cash"])
    parser.add_argument("--architectures", nargs="*", default=None)
    parser.add_argument("--include-hmm-baseline", action="store_true", default=True)
    parser.add_argument("--skip-hmm-baseline", dest="include_hmm_baseline", action="store_false")
    parser.add_argument("--include-random-choice-baseline", action="store_true", default=True)
    parser.add_argument("--skip-random-choice-baseline", dest="include_random_choice_baseline", action="store_false")
    parser.add_argument("--random-choice-seed", type=int, default=42)
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    track_b_dir = Path(__file__).resolve().parent
    data_dir = track_b_dir.parent / "data"
    output_dir = Path(args.output_dir) if args.output_dir else track_b_dir.parent / "artifacts" / f"cluster_action_rankings_k{args.cluster_count}"

    backtest_config = ActionBacktestConfig(
        tradable_assets=tuple(args.assets),
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        risk_free_rate=args.risk_free_rate,
        execution_lag=args.execution_lag,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        objective=args.objective,
        mapping_fit_split=args.mapping_fit_split,
        allow_action_reuse=args.allow_action_reuse,
        cash_proxy_source=args.cash_proxy_source,
    )

    action_library = build_default_action_library(
        n_actions=args.cluster_count,
        tradable_assets=backtest_config.tradable_assets,
    )
    action_library_frame = pd.DataFrame(action_library).T
    action_library_frame.index.name = "action"

    print("Action library:")
    print(action_library_frame.to_string(float_format=lambda value: f"{value:.2f}"))
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

    train_ratio = backtest_config.train_ratio
    experiment_setups = [
        {
            **setup,
            "training_config": replace(
                setup["training_config"],
                train_ratio=train_ratio,
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
        tradable_assets=backtest_config.tradable_assets,
        cash_proxy_source=backtest_config.cash_proxy_source,
        cash_asset_name=backtest_config.cash_asset_name,
    )

    print("Running models:")
    for setup in experiment_setups:
        print(
            f"  - {setup['name']}: architecture={setup['architecture']}, "
            f"train_ratio={setup['training_config'].train_ratio:.2f}"
        )
    print()

    evaluation_results = []
    if args.include_random_choice_baseline:
        random_choice_result = evaluate_random_choice_baseline(
            data_config=returns_data_config,
            asset_returns=asset_returns,
            action_library=action_library,
            config=backtest_config,
            random_state=args.random_choice_seed,
        )
        evaluation_results.append(random_choice_result)
        summary = random_choice_result["summary"]
        print("=== random_choice ===")
        print(
            f"best_mapping={summary['best_mapping']} | "
            f"validation_{args.objective}={summary[f'validation_{args.objective}']:.4f} | "
            f"test_{args.objective}={summary[f'test_{args.objective}']:.4f} | "
            f"combined_{args.objective}={summary[f'combined_{args.objective}']:.4f}"
        )
        print()

    if args.include_hmm_baseline:
        hmm_baseline_result = build_hmm_baseline_experiment_result(
            data_config=returns_data_config,
            target_cluster_count=args.cluster_count,
            random_state=42,
        )
        if hmm_baseline_result is not None:
            evaluation_result = evaluate_model_cluster_actions(
                experiment_result=hmm_baseline_result,
                asset_returns=asset_returns,
                action_library=action_library,
                config=backtest_config,
            )
            evaluation_results.append(evaluation_result)
            summary = evaluation_result["summary"]
            print("=== hmm_baseline ===")
            print(
                f"best_mapping={summary['best_mapping']} | "
                f"validation_{args.objective}={summary[f'validation_{args.objective}']:.4f} | "
                f"test_{args.objective}={summary[f'test_{args.objective}']:.4f} | "
                f"combined_{args.objective}={summary[f'combined_{args.objective}']:.4f}"
            )
            print()
        elif not HMMLEARN_AVAILABLE:
            print("Skipping HMM baseline because hmmlearn is not available.")
            print()

    for setup in experiment_setups:
        runner = runners[setup["architecture"]]
        print(f"=== {setup['name']} ===")
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
        evaluation_result = evaluate_model_cluster_actions(
            experiment_result=experiment_result,
            asset_returns=asset_returns,
            action_library=action_library,
            config=backtest_config,
        )
        evaluation_results.append(evaluation_result)

        summary = evaluation_result["summary"]
        print(
            f"best_mapping={summary['best_mapping']} | "
            f"validation_{args.objective}={summary[f'validation_{args.objective}']:.4f} | "
            f"test_{args.objective}={summary[f'test_{args.objective}']:.4f} | "
            f"combined_{args.objective}={summary[f'combined_{args.objective}']:.4f}"
        )
        print()

    validation_ranking, test_ranking, combined_ranking = build_model_ranking_tables(
        evaluation_results=evaluation_results,
        objective=args.objective,
    )

    save_evaluation_artifacts(
        output_dir=output_dir,
        evaluation_results=evaluation_results,
        validation_ranking=validation_ranking,
        test_ranking=test_ranking,
        combined_ranking=combined_ranking,
    )

    split_summary = evaluation_results[0]["split_summary"] if evaluation_results else pd.DataFrame()

    validation_columns = [
        "validation_rank",
        "experiment_name",
        "architecture",
        f"validation_{args.objective}",
        "validation_annual_return",
        "validation_max_drawdown",
        "best_mapping",
    ]
    test_columns = [
        "test_rank",
        "experiment_name",
        "architecture",
        f"test_{args.objective}",
        "test_annual_return",
        "test_max_drawdown",
        "best_mapping",
    ]
    combined_columns = [
        "combined_rank",
        "experiment_name",
        "architecture",
        f"combined_{args.objective}",
        "combined_annual_return",
        "combined_max_drawdown",
        "best_mapping",
    ]

    if not split_summary.empty:
        print("Dataset split:")
        print(split_summary.to_string(index=False))
        print()

    print("Validation ranking:")
    print(validation_ranking.loc[:, validation_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print("Test ranking:")
    print(test_ranking.loc[:, test_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print("Combined validation+test ranking:")
    print(combined_ranking.loc[:, combined_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(f"Saved ranking artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
