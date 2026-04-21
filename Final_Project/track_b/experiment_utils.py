from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from encoder_only_transformer import (
    ClusteringConfig,
    HMMReferenceConfig,
    SequenceDataConfig,
    TrainingConfig,
    build_split_windows,
    build_hmm_reference,
    build_sequence_panel,
    load_market_and_macro,
    split_ordered_train_random_holdout_indices,
)


def prepare_sequence_experiment_inputs(
    data_config: SequenceDataConfig,
    training_config: TrainingConfig | None = None,
) -> dict:
    training_config = training_config or TrainingConfig()
    market_data, macro_data = load_market_and_macro(data_config)
    sequence_panel, sequence_metadata = build_sequence_panel(market_data, macro_data, data_config)

    row_split_indices = split_ordered_train_random_holdout_indices(
        n_obs=len(sequence_panel),
        train_ratio=training_config.train_ratio,
        validation_ratio=training_config.validation_ratio,
        test_ratio=training_config.test_ratio,
        random_state=training_config.random_state,
    )
    train_feature_frame = sequence_panel.iloc[row_split_indices["train"]].copy()
    sequence_scaler = StandardScaler()
    sequence_scaler.fit(train_feature_frame)
    sequence_z = pd.DataFrame(
        sequence_scaler.transform(sequence_panel),
        index=sequence_panel.index,
        columns=sequence_panel.columns,
    )
    split_windows = build_split_windows(
        frame=sequence_z,
        window_size=data_config.window_size,
        train_ratio=training_config.train_ratio,
        validation_ratio=training_config.validation_ratio,
        test_ratio=training_config.test_ratio,
        random_state=training_config.random_state,
    )
    windows = split_windows["windows"]
    window_end_dates = split_windows["window_end_dates"]

    return {
        "market_data": market_data,
        "macro_data": macro_data,
        "sequence_panel": sequence_panel,
        "sequence_metadata": sequence_metadata,
        "sequence_scaler": sequence_scaler,
        "windows": windows,
        "window_end_dates": window_end_dates,
        "splits": split_windows["splits"],
        "split_summary": split_windows["split_summary"],
        "window_split_indices": split_windows["window_split_indices"],
        "row_split_indices": row_split_indices,
    }


def prepare_hmm_reference_for_experiment(
    data_config: SequenceDataConfig,
    training_config: TrainingConfig,
    hmm_config: HMMReferenceConfig,
    market_data: pd.DataFrame,
    macro_data: pd.DataFrame,
) -> dict | None:
    if not hmm_config.enabled:
        return None

    return build_hmm_reference(
        data_config=data_config,
        market_data=market_data,
        macro_data=macro_data,
        hmm_config=hmm_config,
        random_state=training_config.random_state,
    )


def resolve_target_cluster_count(
    clustering_config: ClusteringConfig,
    hmm_results: dict | None,
) -> int | None:
    if clustering_config.target_cluster_count is not None:
        return int(clustering_config.target_cluster_count)
    if hmm_results is not None:
        return int(hmm_results["best_n_states"])
    return None


def _history_best(history_df: pd.DataFrame | None) -> tuple[int, float]:
    if history_df is None or history_df.empty or "val_loss" not in history_df.columns:
        return 0, np.nan

    valid = history_df.dropna(subset=["val_loss"])
    if valid.empty:
        return 0, np.nan

    best_row = valid.loc[valid["val_loss"].idxmin()]
    return int(best_row.get("epoch", 0)), float(best_row["val_loss"])


def summarize_experiment_outputs(
    experiment_name: str,
    architecture: str,
    sequence_panel: pd.DataFrame,
    sequence_metadata: dict,
    window_size: int,
    window_end_dates: pd.Index,
    history_df: pd.DataFrame | None,
    cluster_scan: pd.DataFrame,
    cluster_labels: np.ndarray,
    target_cluster_count: int,
    hmm_results: dict | None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame | None]:
    silhouette_value = np.nan
    if not cluster_scan.empty and "n_clusters" in cluster_scan.columns and "silhouette" in cluster_scan.columns:
        matched = cluster_scan.loc[cluster_scan["n_clusters"] == target_cluster_count, "silhouette"]
        if not matched.empty:
            silhouette_value = float(matched.iloc[0])

    comparison_table = None
    cluster_summary = (
        pd.DataFrame({"Date": window_end_dates, "cluster": cluster_labels})
        .groupby("cluster")
        .agg(n_windows=("cluster", "size"))
        .reset_index()
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    ari_score = np.nan
    nmi_score = np.nan

    if hmm_results is not None:
        hmm_reference = hmm_results["hmm_reference"]
        aligned_dates = window_end_dates.intersection(hmm_reference.index)
        if len(aligned_dates) > 0:
            positions = pd.Index(window_end_dates).get_indexer(aligned_dates)
            aligned_clusters = pd.Series(cluster_labels[positions], index=aligned_dates, name="cluster")
            aligned_hmm = hmm_reference.loc[aligned_dates, "hmm_state"]
            aligned_hmm_regime = hmm_reference.loc[aligned_dates, "hmm_regime"]

            comparison_table = pd.crosstab(aligned_clusters, aligned_hmm_regime)
            ari_score = float(adjusted_rand_score(aligned_hmm, aligned_clusters))
            nmi_score = float(normalized_mutual_info_score(aligned_hmm, aligned_clusters))

            cluster_profile = pd.DataFrame({"cluster": aligned_clusters, "hmm_regime": aligned_hmm_regime})
            numeric_reference_columns = [
                column
                for column in hmm_reference.columns
                if column not in {"hmm_state", "hmm_regime"} and pd.api.types.is_numeric_dtype(hmm_reference[column])
            ]
            if numeric_reference_columns:
                cluster_profile = cluster_profile.join(hmm_reference.loc[aligned_dates, numeric_reference_columns])

            agg_spec: dict[str, tuple[str, str] | tuple[str, callable]] = {
                "n_windows": ("cluster", "size"),
                "dominant_hmm_regime": ("hmm_regime", lambda series: series.value_counts().index[0]),
            }
            if "SPY_ret" in cluster_profile.columns:
                agg_spec["avg_SPY_ret"] = ("SPY_ret", "mean")
            if "VIX_level" in cluster_profile.columns:
                agg_spec["avg_VIX_level"] = ("VIX_level", "mean")
            if "credit_spread_proxy" in cluster_profile.columns:
                agg_spec["avg_credit_spread_proxy"] = ("credit_spread_proxy", "mean")
            if "curve_slope_10y2y" in cluster_profile.columns:
                agg_spec["avg_curve_slope_10y2y"] = ("curve_slope_10y2y", "mean")

            cluster_summary = cluster_profile.groupby("cluster").agg(**agg_spec).reset_index()

    best_epoch, best_val_loss = _history_best(history_df)
    summary = {
        "experiment_name": experiment_name,
        "architecture": architecture,
        "input_dim": int(sequence_panel.shape[1]),
        "window_size": int(window_size),
        "n_sequence_rows": int(len(sequence_panel)),
        "n_windows": int(len(window_end_dates)),
        "used_sequence_features": ", ".join(sequence_metadata["used_features"]),
        "skipped_sequence_features": ", ".join(sequence_metadata["skipped_features"]),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "target_cluster_count": int(target_cluster_count),
        "silhouette": float(silhouette_value) if not np.isnan(silhouette_value) else np.nan,
        "ari_vs_hmm": ari_score,
        "nmi_vs_hmm": nmi_score,
    }
    return summary, cluster_summary, comparison_table


def build_experiment_result(
    experiment_name: str,
    summary: dict,
    data_config: SequenceDataConfig,
    model_config: object,
    training_config: TrainingConfig | None,
    clustering_config: ClusteringConfig,
    hmm_config: HMMReferenceConfig,
    prepared_inputs: dict,
    model: object,
    history_df: pd.DataFrame | None,
    embeddings: np.ndarray,
    cluster_scan: pd.DataFrame,
    cluster_model: object,
    cluster_labels: np.ndarray,
    cluster_summary: pd.DataFrame,
    comparison_table: pd.DataFrame | None,
    hmm_results: dict | None,
    device: str,
) -> dict:
    return {
        "experiment_name": experiment_name,
        "summary": summary,
        "data_config": data_config,
        "model_config": model_config,
        "training_config": training_config,
        "clustering_config": clustering_config,
        "hmm_config": hmm_config,
        "market_data": prepared_inputs["market_data"],
        "macro_data": prepared_inputs["macro_data"],
        "sequence_panel": prepared_inputs["sequence_panel"],
        "sequence_metadata": prepared_inputs["sequence_metadata"],
        "sequence_scaler": prepared_inputs["sequence_scaler"],
        "windows": prepared_inputs["windows"],
        "window_end_dates": prepared_inputs["window_end_dates"],
        "splits": prepared_inputs["splits"],
        "split_summary": prepared_inputs["split_summary"],
        "window_split_indices": prepared_inputs["window_split_indices"],
        "model": model,
        "history_df": history_df if history_df is not None else pd.DataFrame(),
        "embeddings": embeddings,
        "cluster_scan": cluster_scan,
        "cluster_model": cluster_model,
        "cluster_labels": cluster_labels,
        "cluster_summary": cluster_summary,
        "comparison_table": comparison_table,
        "hmm_results": hmm_results,
        "device": device,
    }


__all__ = [
    "prepare_sequence_experiment_inputs",
    "prepare_hmm_reference_for_experiment",
    "resolve_target_cluster_count",
    "summarize_experiment_outputs",
    "build_experiment_result",
]
