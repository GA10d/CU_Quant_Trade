from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cluster_action_backtest import _metric_sort_key, save_nav_comparison_gif, save_nav_comparison_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge cluster-action and RL backtest outputs into a single NAV comparison plot set.",
    )
    parser.add_argument("--cluster-count", type=int, default=4, choices=[3, 4])
    parser.add_argument("--objective", type=str, default="sharpe")
    parser.add_argument("--cluster-output-dir", type=str, default=None)
    parser.add_argument("--rl-output-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def _load_rankings(result_dir: Path, source_family: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_ranking = pd.read_csv(_require_file(result_dir / "validation_ranking.csv"))
    test_ranking = pd.read_csv(_require_file(result_dir / "test_ranking.csv"))
    split_summary = pd.read_csv(_require_file(result_dir / "split_summary.csv"))

    for frame in (validation_ranking, test_ranking):
        frame["source_family"] = source_family
        frame["raw_experiment_name"] = frame["experiment_name"]
        prefix = "CA" if source_family == "cluster_action" else "RL"
        frame["experiment_name"] = prefix + ":" + frame["experiment_name"].astype(str)

    return validation_ranking, test_ranking, split_summary


def _load_backtest_nav_series(
    result_dir: Path,
    ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    nav_series: dict[str, pd.Series] = {}
    backtests_dir = _require_file(result_dir / "backtests")

    for _, row in ranking_df.iterrows():
        raw_name = str(row["raw_experiment_name"])
        plot_name = str(row["experiment_name"])
        backtest_path = backtests_dir / f"{raw_name}_best_mapping_backtest.csv"
        if not backtest_path.exists():
            continue

        backtest_df = pd.read_csv(backtest_path, parse_dates=["Date"]).set_index("Date")
        if "portfolio_return" not in backtest_df.columns:
            continue

        returns = backtest_df["portfolio_return"].astype(float)
        if returns.empty:
            continue

        nav = (1.0 + returns).cumprod().rename(plot_name)
        nav_series[plot_name] = nav

    if not nav_series:
        return pd.DataFrame()
    return pd.concat(nav_series, axis=1).sort_index().ffill()

def main() -> None:
    args = parse_args()
    track_b_dir = Path(__file__).resolve().parent
    default_cluster_dir = track_b_dir / "results" / f"cluster_action_rankings_k{args.cluster_count}"
    default_rl_dir = track_b_dir / "results" / f"rl_backtest_rankings_k{args.cluster_count}"
    cluster_dir = Path(args.cluster_output_dir) if args.cluster_output_dir else default_cluster_dir
    rl_dir = Path(args.rl_output_dir) if args.rl_output_dir else default_rl_dir
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else track_b_dir / "results" / f"combined_nav_rankings_k{args.cluster_count}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_validation, cluster_test, cluster_split_summary = _load_rankings(
        cluster_dir,
        source_family="cluster_action",
    )
    rl_validation, rl_test, rl_split_summary = _load_rankings(
        rl_dir,
        source_family="rl",
    )

    validation_ranking = pd.concat([cluster_validation, rl_validation], ignore_index=True)
    test_ranking = pd.concat([cluster_test, rl_test], ignore_index=True)

    validation_ranking = validation_ranking.drop(columns=["validation_rank"], errors="ignore")
    test_ranking = test_ranking.drop(columns=["test_rank"], errors="ignore")

    objective_column_validation = f"validation_{args.objective}"
    objective_column_test = f"test_{args.objective}"
    _, descending = _metric_sort_key(args.objective)

    validation_ranking = validation_ranking.sort_values(
        objective_column_validation,
        ascending=not descending,
    ).reset_index(drop=True)
    validation_ranking.insert(0, "validation_rank", range(1, len(validation_ranking) + 1))

    test_ranking = test_ranking.sort_values(
        objective_column_test,
        ascending=not descending,
    ).reset_index(drop=True)
    test_ranking.insert(0, "test_rank", range(1, len(test_ranking) + 1))

    split_summary = cluster_split_summary.copy()
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    validation_ranking.to_csv(output_dir / "combined_validation_ranking.csv", index=False)
    test_ranking.to_csv(output_dir / "combined_test_ranking.csv", index=False)

    full_nav = pd.concat(
        [
            _load_backtest_nav_series(
                cluster_dir,
                validation_ranking[validation_ranking["source_family"] == "cluster_action"],
            ),
            _load_backtest_nav_series(
                rl_dir,
                validation_ranking[validation_ranking["source_family"] == "rl"],
            ),
        ],
        axis=1,
    ).sort_index().ffill()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_order_df = test_ranking.loc[:, ["experiment_name"]].copy()
    save_nav_comparison_png(
        nav_frame=full_nav,
        output_path=plots_dir / "combined_full_period_nav_comparison.png",
        title="Combined NAV Comparison Across Cluster-Action and RL Testbenches",
        ylabel="Normalized NAV",
        ranking_df=plot_order_df,
        split_summary=split_summary,
    )
    gif_saved = save_nav_comparison_gif(
        nav_frame=full_nav,
        output_path=plots_dir / "combined_full_period_nav_comparison.gif",
        title="Combined Full-Period NAV Comparison Animation",
        ylabel="Normalized NAV",
        ranking_df=plot_order_df,
    )

    manifest = pd.DataFrame(
        [
            {"artifact": "combined_full_period_nav_comparison.png", "saved": True},
            {"artifact": "combined_full_period_nav_comparison.gif", "saved": bool(gif_saved)},
        ]
    )
    manifest.to_csv(plots_dir / "plot_manifest.csv", index=False)

    print(f"Cluster-action results dir: {cluster_dir}")
    print(f"RL results dir: {rl_dir}")
    print(f"Saved combined outputs to: {output_dir}")


if __name__ == "__main__":
    main()
