from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from cluster_action_backtest import (
    ActionBacktestConfig,
    build_default_action_library,
    build_hmm_baseline_experiment_result,
    evaluate_random_choice_baseline,
    load_tradable_returns,
)
from encoder_only_transformer import HMMLEARN_AVAILABLE
from experiment_cache import DEFAULT_MODELS_DIR, run_experiment_with_cache
from experiment_presets import build_architecture_runners, build_default_experiment_setups
from rl_backtest_agent import (
    QLearningAgentConfig,
    build_model_ranking_tables,
    evaluate_model_with_q_learning,
    load_q_learning_policy,
    save_rl_ranking_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Q-learning agent on top of Track B cluster states and rank validation/test backtests.",
    )
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
    parser.add_argument("--architectures", nargs="*", default=None)
    parser.add_argument("--include-hmm-baseline", action="store_true", default=True)
    parser.add_argument("--skip-hmm-baseline", dest="include_hmm_baseline", action="store_false")
    parser.add_argument("--include-random-choice-baseline", action="store_true", default=True)
    parser.add_argument("--skip-random-choice-baseline", dest="include_random_choice_baseline", action="store_false")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.985)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--exclude-previous-action-state", action="store_true", default=False)
    parser.add_argument("--reuse-saved-policies", action="store_true", default=False)
    parser.add_argument("--policy-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    track_b_dir = Path(__file__).resolve().parent
    data_dir = track_b_dir.parent / "data"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else track_b_dir.parent / "artifacts" / f"rl_backtest_rankings_k{args.cluster_count}"
    )
    policy_dir = Path(args.policy_dir) if args.policy_dir else output_dir / "q_tables"

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
    agent_config = QLearningAgentConfig(
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        include_previous_action=not args.exclude_previous_action_state,
        eval_interval=args.eval_interval,
        random_state=args.random_state,
        reward_scale=args.reward_scale,
    )

    action_library = build_default_action_library(
        n_actions=args.cluster_count,
        tradable_assets=backtest_config.tradable_assets,
    )
    print("Action library:")
    print(pd.DataFrame(action_library).T.to_string(float_format=lambda value: f"{value:.2f}"))
    print()
    print("Q-learning config:")
    print(
        f"episodes={agent_config.episodes}, alpha={agent_config.learning_rate:.3f}, "
        f"gamma={agent_config.discount_factor:.3f}, epsilon=({agent_config.epsilon_start:.3f}"
        f"->{agent_config.epsilon_end:.3f}, decay={agent_config.epsilon_decay:.3f}), "
        f"include_previous_action={agent_config.include_previous_action}"
    )
    print()
    print(f"reuse_saved_policies={args.reuse_saved_policies}")
    print(f"policy_dir={policy_dir}")
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
                random_state=args.random_state,
            ),
        }
        for setup in experiment_setups
    ]

    returns_data_config = experiment_setups[0]["data_config"]
    asset_returns = load_tradable_returns(
        data_config=returns_data_config,
        tradable_assets=backtest_config.tradable_assets,
    )

    print("Running RL backtests for:")
    for setup in experiment_setups:
        print(f"  - {setup['name']} ({setup['architecture']})")
    print()

    evaluation_results = []
    if args.include_random_choice_baseline:
        random_baseline = evaluate_random_choice_baseline(
            data_config=returns_data_config,
            asset_returns=asset_returns,
            action_library=action_library,
            config=backtest_config,
            random_state=args.random_state,
            experiment_name="random_choice",
        )
        evaluation_results.append(random_baseline)
        summary = random_baseline["summary"]
        print("=== random_choice ===")
        print(
            f"validation_{args.objective}={summary[f'validation_{args.objective}']:.4f} | "
            f"test_{args.objective}={summary[f'test_{args.objective}']:.4f}"
        )
        print()

    if args.include_hmm_baseline:
        hmm_result = build_hmm_baseline_experiment_result(
            data_config=returns_data_config,
            target_cluster_count=args.cluster_count,
            random_state=args.random_state,
        )
        if hmm_result is not None:
            hmm_pretrained_policy = None
            if args.reuse_saved_policies:
                hmm_policy_metadata_path = policy_dir / "hmm_baseline__q_learning_metadata.json"
                if hmm_policy_metadata_path.exists():
                    hmm_pretrained_policy = load_q_learning_policy(hmm_policy_metadata_path)
                    print(f"[policy] loaded saved RL policy from {hmm_policy_metadata_path}")
                else:
                    print(
                        "[policy] no saved RL policy found for hmm_baseline "
                        f"at {hmm_policy_metadata_path}, training a new one."
                    )
            hmm_rl_result = evaluate_model_with_q_learning(
                experiment_result=hmm_result,
                asset_returns=asset_returns,
                action_library=action_library,
                backtest_config=backtest_config,
                agent_config=agent_config,
                pretrained_policy=hmm_pretrained_policy,
            )
            evaluation_results.append(hmm_rl_result)
            summary = hmm_rl_result["summary"]
            print("=== hmm_baseline__q_learning ===")
            print(
                f"validation_{args.objective}={summary[f'validation_{args.objective}']:.4f} | "
                f"test_{args.objective}={summary[f'test_{args.objective}']:.4f} | "
                f"best_episode={summary['best_episode']}"
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
        pretrained_policy = None
        if args.reuse_saved_policies:
            policy_metadata_path = policy_dir / f"{setup['name']}__q_learning_metadata.json"
            if policy_metadata_path.exists():
                pretrained_policy = load_q_learning_policy(policy_metadata_path)
                print(f"[policy] loaded saved RL policy from {policy_metadata_path}")
            else:
                print(f"[policy] no saved RL policy found for {setup['name']} at {policy_metadata_path}, training a new one.")
        rl_result = evaluate_model_with_q_learning(
            experiment_result=experiment_result,
            asset_returns=asset_returns,
            action_library=action_library,
            backtest_config=backtest_config,
            agent_config=agent_config,
            pretrained_policy=pretrained_policy,
        )
        evaluation_results.append(rl_result)

        summary = rl_result["summary"]
        print(
            f"validation_{args.objective}={summary[f'validation_{args.objective}']:.4f} | "
            f"test_{args.objective}={summary[f'test_{args.objective}']:.4f} | "
            f"best_episode={summary['best_episode']}"
        )
        print()

    validation_ranking, test_ranking, combined_ranking = build_model_ranking_tables(
        evaluation_results=evaluation_results,
        objective=args.objective,
    )
    save_rl_ranking_artifacts(
        output_dir=output_dir,
        evaluation_results=evaluation_results,
        validation_ranking=validation_ranking,
        test_ranking=test_ranking,
        combined_ranking=combined_ranking,
    )

    split_summary = evaluation_results[0]["split_summary"] if evaluation_results else pd.DataFrame()
    if not split_summary.empty:
        print("Dataset split:")
        print(split_summary.to_string(index=False))
        print()

    validation_columns = [
        "validation_rank",
        "experiment_name",
        "source_experiment",
        "architecture",
        f"validation_{args.objective}",
        "validation_annual_return",
        "validation_max_drawdown",
        "best_episode",
    ]
    test_columns = [
        "test_rank",
        "experiment_name",
        "source_experiment",
        "architecture",
        f"test_{args.objective}",
        "test_annual_return",
        "test_max_drawdown",
        "best_episode",
    ]

    print("Validation ranking:")
    print(validation_ranking.loc[:, validation_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print("Test ranking:")
    print(test_ranking.loc[:, test_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(f"Saved RL ranking artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
