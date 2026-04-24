from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

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
    tradable_assets: tuple[str, ...] = ("SPY", "TLT", "GLD", "Cash")
    initial_capital: float = 1.0
    transaction_cost: float = 0.001
    risk_free_rate: float = 0.02
    annualization_factor: int = 252
    execution_lag: int = 1
    validation_ratio: float = 0.20
    test_ratio: float = 0.10
    objective: str = "sharpe"
    mapping_fit_split: str = "train"
    allow_action_reuse: bool = False
    fallback_action_name: str | None = None
    split_random_state: int = 42
    cash_proxy_source: str = "FLAT"
    cash_asset_name: str = "Cash"
    exit_on_bankruptcy: bool = False

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
    tradable_assets: tuple[str, ...] = ("SPY", "TLT", "GLD", "Cash"),
) -> dict[str, pd.Series]:
    if n_actions == 3:
        base_templates = {
            "cash": {"Cash": 1.0},
            "defensive": {"SPY": 0.15, "TLT": 0.35, "GLD": 0.20, "Cash": 0.30},
            "aggressive": {"SPY": 0.60, "TLT": 0.15, "GLD": 0.10, "Cash": 0.15},
        }
    elif n_actions == 4:
        base_templates = {
            "cash": {"Cash": 1.0},
            "defensive": {"SPY": 0.10, "TLT": 0.35, "GLD": 0.20, "Cash": 0.35},
            "rebound": {"SPY": 0.45, "TLT": 0.20, "GLD": 0.15, "Cash": 0.20},
            "aggressive": {"SPY": 0.65, "TLT": 0.10, "GLD": 0.10, "Cash": 0.15},
        }
    else:
        raise ValueError("Default action library only supports 3-action or 4-action configurations.")

    return build_action_library_from_templates(
        action_templates=base_templates,
        tradable_assets=tradable_assets,
    )


def build_action_library_from_templates(
    action_templates: dict[str, dict[str, float] | pd.Series],
    tradable_assets: tuple[str, ...] = ("SPY", "TLT", "GLD", "Cash"),
) -> dict[str, pd.Series]:
    if not action_templates:
        raise ValueError("action_templates is empty.")

    action_library: dict[str, pd.Series] = {}
    for action_name, weights in action_templates.items():
        action_weights = pd.Series(weights, dtype=float).reindex(tradable_assets).fillna(0.0)
        total_weight = float(action_weights.sum())
        if total_weight <= 0:
            raise ValueError(f"Action template {action_name} has no overlap with tradable_assets={tradable_assets}.")
        action_library[action_name] = action_weights / total_weight
    return action_library


def load_tradable_returns(
    data_config: SequenceDataConfig,
    tradable_assets: tuple[str, ...],
    cash_proxy_source: str = "FLAT",
    cash_asset_name: str = "Cash",
) -> pd.DataFrame:
    market_data, macro_data = load_market_and_macro(data_config)
    real_assets = [asset for asset in tradable_assets if asset != cash_asset_name]
    missing_assets = [asset for asset in real_assets if asset not in market_data.columns]
    if missing_assets:
        raise ValueError(f"Missing tradable assets in market data: {', '.join(missing_assets)}")

    returns = pd.DataFrame(index=market_data.index)
    if real_assets:
        returns = market_data.loc[:, list(real_assets)].pct_change(fill_method=None)
    if cash_asset_name in tradable_assets:
        returns[cash_asset_name] = load_cash_proxy_returns(
            data_config=data_config,
            target_index=returns.index if not returns.empty else market_data.index,
            macro_data=macro_data,
            cash_proxy_source=cash_proxy_source,
            cash_asset_name=cash_asset_name,
        )
    returns = returns.loc[:, list(tradable_assets)]
    returns = returns.dropna(how="all")
    returns.index.name = "Date"
    return returns


def load_cash_proxy_returns(
    data_config: SequenceDataConfig,
    target_index: pd.Index,
    macro_data: pd.DataFrame | None = None,
    cash_proxy_source: str = "FLAT",
    cash_asset_name: str = "Cash",
) -> pd.Series:
    source = str(cash_proxy_source).upper().strip()
    target_index = pd.Index(pd.to_datetime(target_index), name="Date").sort_values()
    if len(target_index) == 0:
        return pd.Series(dtype=float, name=cash_asset_name)

    if source in {"FLAT", "ZERO", "NONE", "CASH"}:
        return pd.Series(0.0, index=target_index, name=cash_asset_name, dtype=float)

    if source == "RF":
        factor_path = Path(data_config.data_dir) / "factor_data.csv"
        if not factor_path.exists():
            raise FileNotFoundError(f"Cash proxy source RF requires factor_data.csv at {factor_path}.")
        factor_data = pd.read_csv(factor_path, index_col="Date", parse_dates=True).sort_index()
        if "RF" not in factor_data.columns:
            raise KeyError("factor_data.csv does not contain an `RF` column for cash proxy returns.")
        rf_series = pd.to_numeric(factor_data["RF"], errors="coerce") / 100.0
        aligned = (
            rf_series.reindex(rf_series.index.union(target_index).sort_values())
            .ffill()
            .reindex(target_index)
            .fillna(0.0)
        )
        return aligned.rename(cash_asset_name)

    if macro_data is None:
        _, macro_data = load_market_and_macro(data_config)
    if source not in macro_data.columns:
        raise KeyError(f"Cash proxy source {cash_proxy_source!r} not found in macro data columns.")

    annual_rate = pd.to_numeric(macro_data[source], errors="coerce") / 100.0
    annual_rate = annual_rate.sort_index()
    aligned_rate = (
        annual_rate.reindex(annual_rate.index.union(target_index).sort_values())
        .ffill()
        .reindex(target_index)
        .fillna(0.0)
    )
    daily_rate = (1.0 + aligned_rate).pow(1.0 / 252.0) - 1.0
    return daily_rate.rename(cash_asset_name)


def estimate_annual_risk_free_rate(
    macro_data: pd.DataFrame | None,
    dates: pd.Index,
    config: ActionBacktestConfig,
    preferred_columns: tuple[str, ...] = ("DGS3MO", "FEDFUNDS"),
) -> float:
    if macro_data is None or len(dates) == 0:
        return float(config.risk_free_rate)

    macro = macro_data.copy()
    macro.index = pd.to_datetime(macro.index)
    target_dates = pd.Index(pd.to_datetime(dates), name="Date").sort_values()
    percent_rate_columns = {"DGS3MO", "FEDFUNDS", "DGS2", "DGS10"}

    for column in preferred_columns:
        if column not in macro.columns:
            continue

        series = pd.to_numeric(macro[column], errors="coerce").sort_index()
        aligned = (
            series.reindex(series.index.union(target_dates).sort_values())
            .ffill()
            .reindex(target_dates)
            .dropna()
        )
        if aligned.empty:
            continue

        annual_rate = float(aligned.mean())
        if column in percent_rate_columns or abs(annual_rate) > 1.0:
            annual_rate /= 100.0
        return annual_rate

    return float(config.risk_free_rate)


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

    training_config = TrainingConfig(random_state=random_state)
    prepared_inputs = prepare_sequence_experiment_inputs(data_config, training_config=training_config)
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
        "training_config": training_config,
        "clustering_config": None,
        "hmm_config": hmm_config,
        "market_data": prepared_inputs["market_data"],
        "macro_data": prepared_inputs["macro_data"],
        "sequence_panel": prepared_inputs["sequence_panel"],
        "sequence_metadata": prepared_inputs["sequence_metadata"],
        "sequence_scaler": prepared_inputs["sequence_scaler"],
        "windows": prepared_inputs["windows"][positions],
        "window_end_dates": aligned_dates,
        "splits": prepared_inputs["splits"],
        "split_summary": prepared_inputs["split_summary"],
        "window_split_indices": prepared_inputs["window_split_indices"],
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
                    "selection": "ordered_contiguous_block",
                }
            )
    return pd.DataFrame(rows)


def _sorted_unique_ints(values: Iterable) -> list[int]:
    return sorted(int(value) for value in pd.Series(list(values)).dropna().unique())


def _split_date_union(splits: dict[str, pd.Index], split_names: Iterable[str]) -> pd.Index:
    date_parts = [
        pd.Index(splits.get(split_name, pd.Index([], name="Date")), name="Date")
        for split_name in split_names
    ]
    if not date_parts:
        return pd.Index([], name="Date")
    dates = date_parts[0]
    for part in date_parts[1:]:
        dates = dates.union(part)
    return pd.Index(dates.sort_values(), name="Date")


def continuous_span_dates(
    dates: pd.Index,
    available_index: pd.Index | None = None,
) -> pd.Index:
    dates = pd.Index(pd.to_datetime(dates), name="Date").sort_values()
    if len(dates) == 0:
        return pd.Index([], name="Date")
    if available_index is None:
        return dates

    available = pd.Index(pd.to_datetime(available_index), name="Date").sort_values()
    mask = (available >= dates.min()) & (available <= dates.max())
    return pd.Index(available[mask], name="Date")


def combined_validation_test_dates(
    splits: dict[str, pd.Index],
    available_index: pd.Index | None = None,
) -> pd.Index:
    holdout_dates = _split_date_union(splits, ("validation", "test"))
    return continuous_span_dates(holdout_dates, available_index=available_index)


def _infer_target_cluster_count(experiment_result: dict, fallback_cluster_series: pd.Series) -> int:
    summary = experiment_result.get("summary", {})
    for key in ("target_cluster_count", "n_clusters"):
        value = summary.get(key)
        if value is None or pd.isna(value):
            continue
        value = int(value)
        if value > 1:
            return value

    observed = fallback_cluster_series.dropna()
    if observed.empty:
        raise ValueError("Cannot infer target cluster count from empty cluster labels.")
    return int(observed.astype(int).nunique())


def build_leak_free_cluster_series(
    experiment_result: dict,
    splits: dict[str, pd.Index],
    config: ActionBacktestConfig,
    fit_split_names: tuple[str, ...] = ("train", "validation"),
) -> tuple[pd.Series, dict]:
    """Fit evaluation clusters without using test embeddings, then predict all windows.

    Cached experiment results historically include cluster_labels fit on all embeddings.
    The CA/RL testbenches should not use those labels directly, because test windows
    would influence the cluster state definition. This helper refits a lightweight
    MiniBatchKMeans evaluator on train+validation embeddings only.
    """
    window_end_dates = pd.Index(pd.to_datetime(experiment_result["window_end_dates"]), name="Date")
    provided_cluster_series = pd.Series(
        experiment_result["cluster_labels"],
        index=window_end_dates,
        name="cluster",
    )
    target_cluster_count = _infer_target_cluster_count(experiment_result, provided_cluster_series)
    cluster_universe = list(range(target_cluster_count))

    embeddings = experiment_result.get("embeddings")
    if embeddings is None:
        raise ValueError(
            "Experiment result does not contain embeddings. Cannot build leak-free "
            "evaluation clusters without falling back to precomputed cluster_labels."
        )

    embeddings_array = np.asarray(embeddings)
    if embeddings_array.ndim != 2 or len(embeddings_array) != len(window_end_dates):
        raise ValueError(
            "Experiment embeddings must be a 2D array with one row per window. "
            f"Got shape={getattr(embeddings_array, 'shape', None)} for {len(window_end_dates)} windows."
        )
    if not np.isfinite(embeddings_array).all():
        raise ValueError("Experiment embeddings contain non-finite values.")

    fit_dates = _split_date_union(splits, fit_split_names)
    fit_mask = window_end_dates.isin(fit_dates)
    fit_positions = np.flatnonzero(fit_mask)
    if len(fit_positions) < target_cluster_count:
        raise ValueError(
            "Not enough train/validation embeddings to fit a leak-free clusterer: "
            f"{len(fit_positions)} windows for {target_cluster_count} clusters."
        )

    fit_embeddings = embeddings_array[fit_positions]
    clustering_config = experiment_result.get("clustering_config")
    n_init = int(getattr(clustering_config, "n_init", 10))
    batch_size = int(getattr(clustering_config, "batch_size", 1024))
    cluster_model = MiniBatchKMeans(
        n_clusters=target_cluster_count,
        random_state=config.split_random_state,
        n_init=n_init,
        batch_size=batch_size,
    )
    cluster_model.fit(fit_embeddings)
    cluster_labels = cluster_model.predict(embeddings_array)
    cluster_series = pd.Series(cluster_labels.astype(int), index=window_end_dates, name="cluster")

    metadata = {
        "cluster_label_source": "embeddings_fit_on_train_validation",
        "cluster_fit_splits": ",".join(fit_split_names),
        "cluster_fit_n_windows": int(len(fit_positions)),
        "cluster_fit_start": pd.Timestamp(fit_dates.min()) if len(fit_dates) else pd.NaT,
        "cluster_fit_end": pd.Timestamp(fit_dates.max()) if len(fit_dates) else pd.NaT,
        "cluster_model_n_clusters": target_cluster_count,
        "cluster_universe": cluster_universe,
    }
    return cluster_series, metadata


def _fallback_action_name(
    action_library: dict[str, pd.Series],
    explicit_name: str | None,
) -> str:
    if explicit_name is not None:
        if explicit_name not in action_library:
            raise ValueError(f"fallback_action_name={explicit_name!r} is not in the action library.")
        return explicit_name

    for preferred in ("cash", "balanced", "cautious", "rebound", "defensive", "aggressive"):
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


def build_static_weight_frame(
    dates: pd.Index,
    weights: dict[str, float] | pd.Series,
    tradable_assets: tuple[str, ...],
) -> pd.DataFrame:
    dates = pd.Index(pd.to_datetime(dates), name="Date").sort_values()
    if len(dates) == 0:
        raise ValueError("dates is empty.")

    weight_series = pd.Series(weights, dtype=float).reindex(tradable_assets).fillna(0.0)
    total_weight = float(weight_series.sum())
    if total_weight <= 0:
        raise ValueError("Static strategy weights must have a positive total weight.")
    normalized = weight_series / total_weight
    frame = pd.DataFrame(
        np.tile(normalized.to_numpy(dtype=float), (len(dates), 1)),
        index=dates,
        columns=list(tradable_assets),
    )
    frame.index.name = "Date"
    return frame


def _simulate_portfolio_path(
    executed_weights: pd.DataFrame,
    aligned_returns: pd.DataFrame,
    config: ActionBacktestConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, dict]:
    assets = list(executed_weights.columns)
    effective_weights = pd.DataFrame(index=executed_weights.index, columns=assets, dtype=float)
    gross_returns = pd.Series(index=executed_weights.index, dtype=float, name="gross_return")
    transaction_costs = pd.Series(index=executed_weights.index, dtype=float, name="transaction_cost")
    portfolio_returns = pd.Series(index=executed_weights.index, dtype=float, name="portfolio_return")
    portfolio_nav = pd.Series(index=executed_weights.index, dtype=float, name="portfolio_nav")
    turnover = pd.Series(index=executed_weights.index, dtype=float, name="turnover")

    nav = float(config.initial_capital)
    previous_weights = np.zeros(len(assets), dtype=float)
    active = True
    bankruptcy_date = pd.NaT

    for date in executed_weights.index:
        desired_weights = executed_weights.loc[date, assets].to_numpy(dtype=float)
        current_returns = aligned_returns.loc[date, assets].to_numpy(dtype=float)

        if not active:
            effective = np.zeros(len(assets), dtype=float)
            gross = 0.0
            transaction_cost = 0.0
            realized_return = 0.0
        else:
            effective = desired_weights.copy()
            turn = float(np.abs(effective - previous_weights).sum())
            gross = float(np.dot(effective, current_returns))
            transaction_cost = float(turn * config.transaction_cost)
            realized_return = gross - transaction_cost
            if config.exit_on_bankruptcy:
                realized_return = max(realized_return, -1.0)
            nav = float(nav * (1.0 + realized_return))
            if config.exit_on_bankruptcy and nav <= 0.0:
                nav = 0.0
                active = False
                if pd.isna(bankruptcy_date):
                    bankruptcy_date = pd.Timestamp(date)

            previous_weights = np.zeros(len(assets), dtype=float) if not active else effective.copy()
            turnover.loc[date] = turn

        if not active and pd.isna(turnover.loc[date]):
            turnover.loc[date] = 0.0
        effective_weights.loc[date, assets] = effective
        gross_returns.loc[date] = gross
        transaction_costs.loc[date] = transaction_cost
        portfolio_returns.loc[date] = realized_return
        portfolio_nav.loc[date] = nav

    metadata = {
        "bankruptcy_triggered": bool(pd.notna(bankruptcy_date)),
        "bankruptcy_date": bankruptcy_date,
    }
    return (
        effective_weights,
        gross_returns,
        transaction_costs,
        portfolio_returns,
        portfolio_nav,
        turnover,
        metadata,
    )


def run_target_weight_backtest(
    target_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    config: ActionBacktestConfig,
    cluster_series: pd.Series | None = None,
    action_series: pd.Series | None = None,
) -> dict:
    common_dates = pd.Index(target_weights.index).intersection(asset_returns.index)
    if common_dates.empty:
        raise ValueError("No overlapping dates between target_weights and asset_returns.")

    aligned_returns = asset_returns.loc[common_dates, list(config.tradable_assets)].copy().dropna(how="any")
    aligned_target = target_weights.loc[aligned_returns.index, list(config.tradable_assets)].copy().fillna(0.0)
    row_sums = aligned_target.sum(axis=1)
    valid_rows = row_sums > 0
    aligned_target.loc[valid_rows] = aligned_target.loc[valid_rows].div(row_sums[valid_rows], axis=0)

    planned_executed = aligned_target.shift(config.execution_lag).fillna(0.0)
    (
        executed_weights,
        gross_returns,
        transaction_costs,
        portfolio_returns,
        portfolio_nav,
        turnover,
        metadata,
    ) = _simulate_portfolio_path(planned_executed, aligned_returns, config)

    result = {
        "target_weights": aligned_target,
        "executed_weights": executed_weights,
        "asset_returns": aligned_returns,
        "gross_returns": gross_returns,
        "transaction_costs": transaction_costs,
        "portfolio_returns": portfolio_returns,
        "portfolio_nav": portfolio_nav,
        "turnover": turnover,
        **metadata,
    }
    if cluster_series is not None:
        result["cluster_series"] = cluster_series.loc[aligned_returns.index].copy()
    if action_series is not None:
        result["action_series"] = action_series.loc[aligned_returns.index].copy()
    return result


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
    target_weights, action_series = build_weight_frame_from_mapping(
        cluster_series=aligned_clusters,
        action_library=action_library,
        mapping=mapping,
        fallback_action_name=config.fallback_action_name,
    )
    expanded_dates = continuous_span_dates(common_dates, available_index=asset_returns.index)
    target_weights = target_weights.reindex(expanded_dates).ffill().fillna(0.0)
    action_series = action_series.reindex(expanded_dates).ffill().rename("action")
    expanded_clusters = aligned_clusters.reindex(expanded_dates).ffill()
    if expanded_clusters.isna().any():
        raise ValueError("Expanded cluster series still contains NaN values after forward fill.")
    expanded_clusters = expanded_clusters.astype(int).rename("cluster")
    return run_target_weight_backtest(
        target_weights=target_weights,
        asset_returns=asset_returns,
        config=config,
        cluster_series=expanded_clusters,
        action_series=action_series,
    )


def run_static_weight_backtest(
    strategy_name: str,
    weights: dict[str, float] | pd.Series,
    asset_returns: pd.DataFrame,
    config: ActionBacktestConfig,
) -> dict:
    target_weights = build_static_weight_frame(
        dates=asset_returns.index,
        weights=weights,
        tradable_assets=config.tradable_assets,
    )
    action_series = pd.Series(strategy_name, index=target_weights.index, name="action", dtype="object")
    cluster_series = pd.Series(0, index=target_weights.index, name="cluster", dtype=int)
    return run_target_weight_backtest(
        target_weights=target_weights,
        asset_returns=asset_returns,
        config=config,
        cluster_series=cluster_series,
        action_series=action_series,
    )


def evaluate_static_weight_strategy(
    strategy_name: str,
    weights: dict[str, float] | pd.Series,
    asset_returns: pd.DataFrame,
    config: ActionBacktestConfig,
    annual_risk_free_rate: float | None = None,
) -> dict:
    backtest_result = run_static_weight_backtest(
        strategy_name=strategy_name,
        weights=weights,
        asset_returns=asset_returns,
        config=config,
    )
    metrics = calculate_backtest_metrics(
        portfolio_returns=backtest_result["portfolio_returns"],
        turnover=backtest_result["turnover"],
        config=config,
        risk_free_rate=annual_risk_free_rate,
    )
    summary = {
        "strategy_name": strategy_name,
        "initial_capital": float(config.initial_capital),
        "exit_on_bankruptcy": bool(config.exit_on_bankruptcy),
        "bankruptcy_triggered": bool(backtest_result.get("bankruptcy_triggered", False)),
        "bankruptcy_date": backtest_result.get("bankruptcy_date"),
    }
    summary.update(metrics)
    return {
        "summary": summary,
        "backtest_result": backtest_result,
        "metrics": metrics,
        "weights": pd.Series(weights, dtype=float).reindex(config.tradable_assets).fillna(0.0),
    }


def calculate_backtest_metrics(
    portfolio_returns: pd.Series,
    turnover: pd.Series,
    config: ActionBacktestConfig,
    risk_free_rate: float | None = None,
) -> dict:
    returns = portfolio_returns.dropna()
    turnover = turnover.loc[returns.index].fillna(0.0)
    annual_risk_free_rate = float(config.risk_free_rate if risk_free_rate is None else risk_free_rate)
    if returns.empty:
        return {
            "n_days": 0,
            "risk_free_rate": annual_risk_free_rate,
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
        sharpe = float((annual_return - annual_risk_free_rate) / annual_vol)
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
        "risk_free_rate": annual_risk_free_rate,
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
    fit_dates: pd.Index,
    config: ActionBacktestConfig,
    fit_split_name: str,
    fit_risk_free_rate: float | None = None,
) -> dict:
    fit_clusters = cluster_series.loc[cluster_series.index.intersection(fit_dates)].dropna()
    if fit_clusters.empty:
        raise ValueError(f"{fit_split_name} cluster series is empty after alignment.")

    candidate_cluster_ids = _sorted_unique_ints(fit_clusters)
    candidate_mappings = enumerate_cluster_action_mappings(
        cluster_ids=candidate_cluster_ids,
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
        fit_returns = backtest_result["portfolio_returns"].loc[
            backtest_result["portfolio_returns"].index.intersection(fit_dates)
        ]
        fit_turnover = backtest_result["turnover"].loc[fit_returns.index]
        fit_metrics = calculate_backtest_metrics(
            fit_returns,
            fit_turnover,
            config,
            risk_free_rate=fit_risk_free_rate,
        )
        objective_value = _objective_value(fit_metrics, config.objective)

        row = {
            "mapping": serialize_mapping(mapping),
            "objective_value": objective_value,
        }
        row.update({f"{fit_split_name}_{key}": value for key, value in fit_metrics.items()})
        search_rows.append(row)

        if objective_value > best_score:
            best_score = objective_value
            best_mapping = mapping
            best_backtest = backtest_result
            best_metrics = fit_metrics

    assert best_mapping is not None
    assert best_backtest is not None
    assert best_metrics is not None

    search_df = pd.DataFrame(search_rows).sort_values("objective_value", ascending=False).reset_index(drop=True)
    return {
        "best_mapping": best_mapping,
        "best_mapping_text": serialize_mapping(best_mapping),
        "best_score": best_score,
        "candidate_cluster_ids": candidate_cluster_ids,
        "fit_split_name": fit_split_name,
        "fit_metrics": best_metrics,
        "search_results": search_df,
        "backtest_result": best_backtest,
    }


def evaluate_model_cluster_actions(
    experiment_result: dict,
    asset_returns: pd.DataFrame,
    action_library: dict[str, pd.Series],
    config: ActionBacktestConfig,
) -> dict:
    window_end_dates = pd.Index(pd.to_datetime(experiment_result["window_end_dates"]), name="Date")
    if "splits" in experiment_result and experiment_result["splits"] is not None:
        splits = {
            split_name: pd.Index(split_dates, name="Date")
            for split_name, split_dates in experiment_result["splits"].items()
        }
        split_summary = experiment_result.get("split_summary", summarize_split_dates(splits))
    else:
        splits = split_window_dates(
            window_end_dates=window_end_dates,
            validation_ratio=config.validation_ratio,
            test_ratio=config.test_ratio,
            random_state=config.split_random_state,
        )
        split_summary = summarize_split_dates(splits)

    cluster_series, cluster_metadata = build_leak_free_cluster_series(
        experiment_result=experiment_result,
        splits=splits,
        config=config,
    )
    macro_data = experiment_result.get("macro_data")
    train_risk_free_rate = estimate_annual_risk_free_rate(macro_data, splits["train"], config)
    validation_dates = combined_validation_test_dates(splits, available_index=asset_returns.index)
    validation_risk_free_rate = estimate_annual_risk_free_rate(macro_data, validation_dates, config)
    test_risk_free_rate = validation_risk_free_rate
    combined_dates = validation_dates
    combined_risk_free_rate = validation_risk_free_rate
    mapping_fit_split = config.mapping_fit_split.lower().strip()
    allowed_mapping_fit_splits = {"train", "validation"}
    if mapping_fit_split not in allowed_mapping_fit_splits:
        raise ValueError(
            f"mapping_fit_split={config.mapping_fit_split!r} is not one of: "
            f"{', '.join(sorted(allowed_mapping_fit_splits))}."
        )
    split_risk_free_rates = {
        "train": train_risk_free_rate,
        "validation": validation_risk_free_rate,
        "test": test_risk_free_rate,
    }

    search_result = search_best_cluster_action_mapping(
        cluster_series=cluster_series,
        asset_returns=asset_returns,
        action_library=action_library,
        fit_dates=splits[mapping_fit_split],
        config=config,
        fit_split_name=mapping_fit_split,
        fit_risk_free_rate=split_risk_free_rates[mapping_fit_split],
    )
    best_backtest = search_result["backtest_result"]

    combined_backtest = run_action_mapping_backtest(
        cluster_series=cluster_series.loc[cluster_series.index.intersection(combined_dates)],
        asset_returns=asset_returns,
        action_library=action_library,
        mapping=search_result["best_mapping"],
        config=config,
    )
    combined_returns = combined_backtest["portfolio_returns"]
    combined_turnover = combined_backtest["turnover"]
    combined_metrics = calculate_backtest_metrics(
        combined_returns,
        combined_turnover,
        config,
        risk_free_rate=combined_risk_free_rate,
    )
    validation_metrics = dict(combined_metrics)
    test_metrics = dict(combined_metrics)
    holdout_cluster_dates = combined_validation_test_dates(splits)
    validation_cluster_ids = _sorted_unique_ints(
        cluster_series.loc[cluster_series.index.intersection(holdout_cluster_dates)]
    )
    test_cluster_ids = _sorted_unique_ints(
        cluster_series.loc[cluster_series.index.intersection(splits["test"])]
    )
    mapping_fit_cluster_ids = _sorted_unique_ints(
        cluster_series.loc[cluster_series.index.intersection(splits[mapping_fit_split])]
    )
    mapped_cluster_ids = _sorted_unique_ints(search_result["best_mapping"].keys())
    unmapped_validation_clusters = sorted(set(validation_cluster_ids) - set(mapped_cluster_ids))
    unmapped_test_clusters = sorted(set(test_cluster_ids) - set(mapped_cluster_ids))

    summary = {
        "experiment_name": experiment_result["experiment_name"],
        "architecture": experiment_result["summary"].get("architecture"),
        "n_clusters": int(np.unique(cluster_series.values).size),
        "target_cluster_count": int(cluster_metadata["cluster_model_n_clusters"]),
        "best_mapping": search_result["best_mapping_text"],
        "objective": config.objective,
        "cluster_label_source": cluster_metadata["cluster_label_source"],
        "cluster_fit_splits": cluster_metadata["cluster_fit_splits"],
        "cluster_fit_n_windows": cluster_metadata["cluster_fit_n_windows"],
        "mapping_fit_split": mapping_fit_split,
        "mapping_candidate_clusters": ",".join(map(str, mapping_fit_cluster_ids)),
        "validation_clusters": ",".join(map(str, validation_cluster_ids)),
        "unmapped_validation_clusters": ",".join(map(str, unmapped_validation_clusters)),
        "unmapped_test_clusters": ",".join(map(str, unmapped_test_clusters)),
    }
    summary.update({f"validation_{key}": value for key, value in validation_metrics.items()})
    summary.update({f"test_{key}": value for key, value in test_metrics.items()})
    summary.update({f"combined_{key}": value for key, value in combined_metrics.items()})

    return {
        "summary": summary,
        "splits": splits,
        "split_summary": split_summary,
        "best_mapping": search_result["best_mapping"],
        "best_mapping_text": search_result["best_mapping_text"],
        "search_results": search_result["search_results"],
        "backtest_result": best_backtest,
        "combined_backtest_result": combined_backtest,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "combined_metrics": combined_metrics,
        "cluster_metadata": cluster_metadata,
    }


def evaluate_random_choice_baseline(
    data_config: SequenceDataConfig,
    asset_returns: pd.DataFrame,
    action_library: dict[str, pd.Series],
    config: ActionBacktestConfig,
    random_state: int = 42,
    experiment_name: str = "random_choice",
) -> dict:
    prepared_inputs = prepare_sequence_experiment_inputs(
        data_config,
        training_config=TrainingConfig(
            train_ratio=config.train_ratio,
            validation_ratio=config.validation_ratio,
            test_ratio=config.test_ratio,
            random_state=config.split_random_state,
        ),
    )
    window_end_dates = pd.Index(prepared_inputs["window_end_dates"], name="Date")
    splits = prepared_inputs["splits"]
    split_summary = prepared_inputs["split_summary"]
    macro_data = prepared_inputs.get("macro_data")
    validation_dates = combined_validation_test_dates(splits, available_index=asset_returns.index)
    validation_risk_free_rate = estimate_annual_risk_free_rate(macro_data, validation_dates, config)
    test_risk_free_rate = validation_risk_free_rate
    combined_dates = validation_dates
    combined_risk_free_rate = validation_risk_free_rate

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

    combined_backtest = run_action_mapping_backtest(
        cluster_series=cluster_series.loc[cluster_series.index.intersection(combined_dates)],
        asset_returns=asset_returns,
        action_library=action_library,
        mapping=mapping,
        config=config,
    )
    combined_returns = combined_backtest["portfolio_returns"]
    combined_turnover = combined_backtest["turnover"]
    combined_metrics = calculate_backtest_metrics(
        combined_returns,
        combined_turnover,
        config,
        risk_free_rate=combined_risk_free_rate,
    )
    validation_metrics = dict(combined_metrics)
    test_metrics = dict(combined_metrics)

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
    summary.update({f"combined_{key}": value for key, value in combined_metrics.items()})

    return {
        "summary": summary,
        "splits": splits,
        "split_summary": split_summary,
        "best_mapping": mapping,
        "best_mapping_text": best_mapping_text,
        "search_results": search_results,
        "backtest_result": backtest_result,
        "combined_backtest_result": combined_backtest,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "combined_metrics": combined_metrics,
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
    combined_column = f"combined_{objective}"

    validation_ranking = summary_df.sort_values(validation_column, ascending=not descending).reset_index(drop=True)
    validation_ranking.insert(0, "validation_rank", np.arange(1, len(validation_ranking) + 1))

    test_ranking = summary_df.sort_values(test_column, ascending=not descending).reset_index(drop=True)
    test_ranking.insert(0, "test_rank", np.arange(1, len(test_ranking) + 1))

    combined_ranking = summary_df.sort_values(combined_column, ascending=not descending).reset_index(drop=True)
    combined_ranking.insert(0, "combined_rank", np.arange(1, len(combined_ranking) + 1))
    return validation_ranking, test_ranking, combined_ranking


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
        split_rows = []
        for split_name, split_dates in evaluation_results[0]["splits"].items():
            for split_date in pd.Index(split_dates):
                split_rows.append({"Date": pd.Timestamp(split_date), "split": split_name})
        if split_rows:
            split_dates_frame = pd.DataFrame(split_rows).sort_values(["Date", "split"]).reset_index(drop=True)
            split_dates_frame.to_csv(output_path / "split_dates.csv", index=False)

    search_dir = output_path / "mapping_search"
    backtest_dir = output_path / "backtests"
    search_dir.mkdir(exist_ok=True)
    backtest_dir.mkdir(exist_ok=True)

    for result in evaluation_results:
        experiment_name = result["summary"]["experiment_name"]
        safe_name = experiment_name.replace("/", "_")
        mapping_fit_split = result["summary"].get("mapping_fit_split", "validation")
        result["search_results"].to_csv(search_dir / f"{safe_name}_{mapping_fit_split}_search.csv", index=False)

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
        start_date = pd.to_datetime(row.get("start_date"), errors="coerce")
        end_date = pd.to_datetime(row.get("end_date"), errors="coerce")
        if pd.isna(start_date) or pd.isna(end_date):
            continue
        if row.get("selection") not in {"ordered_oldest", "ordered_contiguous_block"}:
            continue
        ax.axvspan(start_date, end_date, alpha=0.12, color=color_map.get(row["split"], "#eeeeee"))

    for _, row in split_summary.iterrows():
        start_date = pd.to_datetime(row.get("start_date"), errors="coerce")
        if pd.isna(start_date):
            continue
        if row.get("selection") not in {"ordered_oldest", "ordered_contiguous_block"}:
            continue
        if row["split"] == "train":
            continue
        ax.axvline(start_date, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(
            start_date,
            y_max,
            f" {row['split']}",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color="gray",
        )


def _nav_plot_size(n_series: int) -> tuple[float, float]:
    n = max(int(n_series), 1)
    width = min(28.0, max(16.0, 16.0 + 0.55 * max(0, n - 8)))
    height = min(16.0, max(8.0, 8.0 + 0.22 * max(0, n - 10)))
    return width, height


def _nav_gif_layout(n_series: int) -> tuple[tuple[float, float], list[float]]:
    n = max(int(n_series), 1)
    width = min(32.0, max(20.0, 20.0 + 0.7 * max(0, n - 8)))
    height = min(18.0, max(9.0, 9.0 + 0.26 * max(0, n - 10)))
    rank_panel_width = min(3.4, max(2.0, 2.0 + 0.08 * max(0, n - 8)))
    return (width, height), [5.0, rank_panel_width]


def save_nav_comparison_png(
    nav_frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
    ylabel: str,
    ranking_df: pd.DataFrame | None = None,
    split_summary: pd.DataFrame | None = None,
    line_colors: dict[str, object] | None = None,
) -> None:
    if nav_frame.empty:
        return

    ordered_columns = list(nav_frame.columns)
    if ranking_df is not None and not ranking_df.empty and "experiment_name" in ranking_df.columns:
        ranked = [name for name in ranking_df["experiment_name"] if name in nav_frame.columns]
        ordered_columns = ranked + [name for name in nav_frame.columns if name not in ranked]

    plot_frame = nav_frame.loc[:, ordered_columns]
    fig, ax = plt.subplots(figsize=_nav_plot_size(len(plot_frame.columns)))
    for column in plot_frame.columns:
        plot_kwargs = {"linewidth": 1.8, "label": column}
        if line_colors is not None and column in line_colors:
            plot_kwargs["color"] = line_colors[column]
        ax.plot(plot_frame.index, plot_frame[column], **plot_kwargs)

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
    line_colors: dict[str, object] | None = None,
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

    figure_size, width_ratios = _nav_gif_layout(len(plot_frame.columns))
    fig, (ax, ax_rank) = plt.subplots(
        ncols=2,
        figsize=figure_size,
        gridspec_kw={"width_ratios": width_ratios},
    )
    lines = {}
    for column in plot_frame.columns:
        plot_kwargs = {"linewidth": 2.0, "label": column}
        if line_colors is not None and column in line_colors:
            plot_kwargs["color"] = line_colors[column]
        lines[column] = ax.plot([], [], **plot_kwargs)[0]
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
    "build_action_library_from_templates",
    "load_tradable_returns",
    "load_cash_proxy_returns",
    "estimate_annual_risk_free_rate",
    "combined_validation_test_dates",
    "continuous_span_dates",
    "build_hmm_baseline_experiment_result",
    "split_window_dates",
    "summarize_split_dates",
    "build_leak_free_cluster_series",
    "enumerate_cluster_action_mappings",
    "serialize_mapping",
    "build_weight_frame_from_mapping",
    "build_static_weight_frame",
    "run_target_weight_backtest",
    "run_action_mapping_backtest",
    "run_static_weight_backtest",
    "evaluate_static_weight_strategy",
    "calculate_backtest_metrics",
    "search_best_cluster_action_mapping",
    "evaluate_model_cluster_actions",
    "evaluate_random_choice_baseline",
    "build_model_ranking_tables",
    "save_evaluation_artifacts",
]
