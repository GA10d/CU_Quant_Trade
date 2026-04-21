from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_classif
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TRACK_C_DIR = Path(__file__).resolve().parent
TRACK_B_DIR = TRACK_C_DIR.parent / "track_b"
if str(TRACK_B_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_B_DIR))


from encoder_only_transformer import (  # noqa: E402
    ClusteringConfig,
    HMMReferenceConfig,
    SequenceDataConfig,
    TrainingConfig,
    build_split_windows,
    cluster_embeddings,
    load_market_and_macro,
    load_market_tuple_panels,
    make_torch_dataloader_generator,
    resolve_device,
    set_seed,
    split_ordered_train_random_holdout_indices,
)
from experiment_utils import (  # noqa: E402
    build_experiment_result,
    prepare_hmm_reference_for_experiment,
    summarize_experiment_outputs,
)


@dataclass
class EventEncodingConfig:
    quantile_lookback: int = 63
    breakout_lookback: int = 20
    volatility_lookback: int = 20
    volatility_quantile_lookback: int = 126
    volume_zscore_lookback: int = 63
    upper_quantile: float = 0.80
    lower_quantile: float = 0.20
    volume_spike_z: float = 2.0
    volume_dryup_z: float = -1.0
    include_cross_asset_events: bool = True


@dataclass
class EventSNNConfig:
    architecture: str = "snn_event_lif"
    hidden_dim: int = 96
    embedding_dim: int = 48
    beta: float = 0.90
    threshold: float = 1.0
    dropout: float = 0.10
    recurrent_scale: float = 0.25
    use_mask_embedding: bool = True


@dataclass
class EventSelectionConfig:
    enabled: bool = False
    min_activation_rate: float = 0.0
    max_activation_rate: float = 0.35
    max_features: int | None = None
    min_mutual_info: float | None = None
    preserve_cross_asset_events: bool = True
    target_assets: tuple[str, ...] = ("SPY", "TLT", "GLD", "VIX")


def build_default_track_c_data_config(data_dir: str | Path) -> SequenceDataConfig:
    return SequenceDataConfig(
        data_dir=Path(data_dir),
        market_filename="market_data_full_adjusted.csv",
        window_size=60,
        sequence_feature_mode="market_tuples",
        tuple_asset_columns=("GLD", "HYG", "LQD", "SPY", "TLT", "UUP", "VIX"),
        tuple_market_fields=("Close", "High", "Low", "Open", "Volume"),
        tuple_log1p_volume=True,
        include_vix_log_return=False,
        include_curve_slope_change=False,
        strict_feature_availability=False,
    )


def build_default_event_training_config(device: str = "auto") -> TrainingConfig:
    return TrainingConfig(
        mask_ratio=0.15,
        batch_size=64,
        num_epochs=45,
        learning_rate=2e-3,
        weight_decay=1e-4,
        train_ratio=0.7,
        validation_ratio=0.2,
        test_ratio=0.1,
        early_stopping_patience=8,
        min_epochs=6,
        random_state=42,
        log_every=1,
        device=device,
    )


def build_default_event_model_config() -> EventSNNConfig:
    return EventSNNConfig()


def _rolling_quantile(series: pd.Series, window: int, quantile: float) -> pd.Series:
    min_periods = max(10, window // 3)
    return series.rolling(window=window, min_periods=min_periods).quantile(quantile).shift(1)


def _rolling_mean_std(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    min_periods = max(10, window // 3)
    mean = series.rolling(window=window, min_periods=min_periods).mean().shift(1)
    std = series.rolling(window=window, min_periods=min_periods).std(ddof=0).shift(1)
    std = std.where(std > 1e-8)
    return mean, std


def _to_event(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(np.float32)


def _extract_asset_event_features(
    asset: str,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    open_: pd.Series,
    volume: pd.Series,
    event_config: EventEncodingConfig,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    prev_close = close.shift(1)
    log_ret = np.log(close.clip(lower=1e-8)).diff()
    gap = (open_ - prev_close) / prev_close.replace(0.0, np.nan)
    intraday_range = (high - low) / prev_close.replace(0.0, np.nan)
    log_volume = np.log1p(volume.clip(lower=0.0))

    ret_up_threshold = _rolling_quantile(log_ret, event_config.quantile_lookback, event_config.upper_quantile)
    ret_down_threshold = _rolling_quantile(log_ret, event_config.quantile_lookback, event_config.lower_quantile)

    gap_up_threshold = _rolling_quantile(gap, event_config.quantile_lookback, event_config.upper_quantile)
    gap_down_threshold = _rolling_quantile(gap, event_config.quantile_lookback, event_config.lower_quantile)

    range_expand_threshold = _rolling_quantile(intraday_range, event_config.quantile_lookback, event_config.upper_quantile)

    volume_mean, volume_std = _rolling_mean_std(log_volume, event_config.volume_zscore_lookback)
    volume_zscore = (log_volume - volume_mean) / volume_std

    breakout_high = close.rolling(
        window=event_config.breakout_lookback,
        min_periods=max(5, event_config.breakout_lookback // 2),
    ).max().shift(1)
    breakdown_low = close.rolling(
        window=event_config.breakout_lookback,
        min_periods=max(5, event_config.breakout_lookback // 2),
    ).min().shift(1)

    realized_vol = log_ret.rolling(
        window=event_config.volatility_lookback,
        min_periods=max(5, event_config.volatility_lookback // 2),
    ).std(ddof=0)
    vol_up_threshold = _rolling_quantile(
        realized_vol,
        event_config.volatility_quantile_lookback,
        event_config.upper_quantile,
    )
    vol_down_threshold = _rolling_quantile(
        realized_vol,
        event_config.volatility_quantile_lookback,
        event_config.lower_quantile,
    )

    event_columns = {
        f"{asset}_ret_up_large": _to_event(log_ret > ret_up_threshold),
        f"{asset}_ret_down_large": _to_event(log_ret < ret_down_threshold),
        f"{asset}_gap_up": _to_event(gap > gap_up_threshold),
        f"{asset}_gap_down": _to_event(gap < gap_down_threshold),
        f"{asset}_range_expand": _to_event(intraday_range > range_expand_threshold),
        f"{asset}_volume_spike": _to_event(volume_zscore > event_config.volume_spike_z),
        f"{asset}_volume_dryup": _to_event(volume_zscore < event_config.volume_dryup_z),
        f"{asset}_breakout_{event_config.breakout_lookback}d": _to_event(close > breakout_high),
        f"{asset}_breakdown_{event_config.breakout_lookback}d": _to_event(close < breakdown_low),
        f"{asset}_vol_regime_up": _to_event(realized_vol > vol_up_threshold),
        f"{asset}_vol_regime_down": _to_event(realized_vol < vol_down_threshold),
    }
    derived_series = {
        "log_ret": log_ret,
        "gap": gap,
        "intraday_range": intraday_range,
        "volume_zscore": volume_zscore,
        "realized_vol": realized_vol,
        "close": close,
    }
    return event_columns, derived_series


def build_event_panel(
    data_config: SequenceDataConfig,
    event_config: EventEncodingConfig,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    market_data, macro_data = load_market_and_macro(data_config)
    tuple_panels = load_market_tuple_panels(data_config, target_index=market_data.index)
    close_panel = tuple_panels["Close"].copy()
    high_panel = tuple_panels["High"].copy()
    low_panel = tuple_panels["Low"].copy()
    open_panel = tuple_panels["Open"].copy()
    volume_panel = tuple_panels["Volume"].copy()

    event_columns: dict[str, pd.Series] = {}
    derived_features: dict[str, dict[str, pd.Series]] = {}

    for asset in data_config.tuple_asset_columns:
        if asset not in close_panel.columns:
            continue
        asset_events, asset_derived = _extract_asset_event_features(
            asset=asset,
            close=close_panel[asset].copy(),
            high=high_panel[asset].copy(),
            low=low_panel[asset].copy(),
            open_=open_panel[asset].copy(),
            volume=volume_panel[asset].copy(),
            event_config=event_config,
        )
        event_columns.update(asset_events)
        derived_features[asset] = asset_derived

    if event_config.include_cross_asset_events:
        if {"SPY", "TLT", "VIX"}.issubset(derived_features):
            event_columns["risk_off_classic"] = _to_event(
                (event_columns["SPY_ret_down_large"] > 0)
                & (event_columns["VIX_ret_up_large"] > 0)
                & (event_columns["TLT_ret_up_large"] > 0)
            )
            event_columns["flight_to_safety"] = _to_event(
                (event_columns["SPY_ret_down_large"] > 0)
                & (event_columns["TLT_ret_up_large"] > 0)
            )
            event_columns["risk_on_rebound"] = _to_event(
                (event_columns["SPY_ret_up_large"] > 0)
                & (event_columns["VIX_ret_down_large"] > 0)
                & (event_columns["TLT_ret_down_large"] > 0)
            )

        if {"HYG", "LQD"}.issubset(derived_features):
            credit_spread = derived_features["HYG"]["log_ret"] - derived_features["LQD"]["log_ret"]
            credit_stress_threshold = _rolling_quantile(
                credit_spread,
                event_config.quantile_lookback,
                event_config.lower_quantile,
            )
            event_columns["credit_stress"] = _to_event(
                (event_columns["HYG_ret_down_large"] > 0)
                & (credit_spread < credit_stress_threshold)
            )

        if {"UUP", "SPY"}.issubset(derived_features):
            event_columns["usd_stress"] = _to_event(
                (event_columns["UUP_ret_up_large"] > 0)
                & (event_columns["SPY_ret_down_large"] > 0)
            )

        if {"GLD", "SPY"}.issubset(derived_features):
            event_columns["gold_safe_haven"] = _to_event(
                (event_columns["GLD_ret_up_large"] > 0)
                & (event_columns["SPY_ret_down_large"] > 0)
            )

    event_panel = pd.concat(event_columns.values(), axis=1)
    event_panel.columns = list(event_columns.keys())
    event_panel.index.name = "Date"

    warmup = max(
        event_config.quantile_lookback,
        event_config.breakout_lookback,
        event_config.volatility_lookback,
        event_config.volatility_quantile_lookback,
        event_config.volume_zscore_lookback,
    ) + 2
    if len(event_panel) <= warmup:
        raise ValueError("Event panel is shorter than the configured warmup period.")

    event_panel = event_panel.iloc[warmup:].copy()
    event_panel = event_panel.loc[event_panel.index.intersection(market_data.index)].sort_index()
    event_panel = event_panel.astype(np.float32)

    inactive_columns = [column for column in event_panel.columns if float(event_panel[column].sum()) <= 0.0]
    if inactive_columns:
        event_panel = event_panel.drop(columns=inactive_columns)

    if event_panel.empty:
        raise ValueError("Event panel is empty after warmup and inactive-column filtering.")

    metadata = {
        "used_features": list(event_panel.columns),
        "skipped_features": inactive_columns,
        "warmup_rows": int(warmup),
        "event_density": float(event_panel.to_numpy(dtype=np.float32).mean()),
        "selection_applied": False,
    }
    return event_panel, metadata, market_data, macro_data


def _build_feature_selection_targets(
    market_data: pd.DataFrame,
    event_index: pd.Index,
    target_assets: tuple[str, ...],
) -> pd.DataFrame:
    next_day_returns = market_data.loc[:, [asset for asset in target_assets if asset in market_data.columns]].pct_change(
        fill_method=None
    ).shift(-1)
    target_frame = pd.DataFrame(index=event_index)
    if "SPY" in next_day_returns.columns:
        target_frame["spy_up_next"] = (next_day_returns["SPY"] > 0.0).astype(float)
    if "TLT" in next_day_returns.columns:
        target_frame["tlt_up_next"] = (next_day_returns["TLT"] > 0.0).astype(float)
    if "GLD" in next_day_returns.columns:
        target_frame["gld_up_next"] = (next_day_returns["GLD"] > 0.0).astype(float)
    if {"SPY", "TLT", "VIX"}.issubset(next_day_returns.columns):
        target_frame["risk_off_next"] = (
            (next_day_returns["SPY"] < 0.0)
            & (next_day_returns["TLT"] > 0.0)
            & (next_day_returns["VIX"] > 0.0)
        ).astype(float)
    return target_frame.reindex(event_index)


def score_event_features(
    event_panel: pd.DataFrame,
    market_data: pd.DataFrame,
    training_config: TrainingConfig,
    selection_config: EventSelectionConfig,
    cross_asset_feature_names: set[str] | None = None,
) -> pd.DataFrame:
    if event_panel.empty:
        raise ValueError("event_panel is empty.")

    split_indices = split_ordered_train_random_holdout_indices(
        n_obs=len(event_panel),
        train_ratio=training_config.train_ratio,
        validation_ratio=training_config.validation_ratio,
        test_ratio=training_config.test_ratio,
        random_state=training_config.random_state,
    )
    train_panel = event_panel.iloc[split_indices["train"]].copy()
    activation_rate = train_panel.mean(axis=0)

    target_frame = _build_feature_selection_targets(
        market_data=market_data,
        event_index=event_panel.index,
        target_assets=selection_config.target_assets,
    ).iloc[split_indices["train"]]

    score_frame = pd.DataFrame(index=event_panel.columns)
    score_frame["train_activation_rate"] = activation_rate
    score_frame["mutual_info_mean"] = 0.0
    score_frame["target_count"] = 0

    X = train_panel.astype(int)
    valid_target_count = 0
    for target_name in target_frame.columns:
        y = target_frame[target_name].dropna()
        if y.empty or y.nunique() < 2:
            continue

        aligned_index = X.index.intersection(y.index)
        if len(aligned_index) < 50:
            continue

        mi_values = mutual_info_classif(
            X.loc[aligned_index],
            y.loc[aligned_index].astype(int),
            discrete_features=True,
            random_state=training_config.random_state,
        )
        score_frame[target_name] = mi_values
        score_frame["mutual_info_mean"] += mi_values
        valid_target_count += 1

    if valid_target_count > 0:
        score_frame["mutual_info_mean"] = score_frame["mutual_info_mean"] / float(valid_target_count)
    score_frame["target_count"] = valid_target_count
    score_frame["is_cross_asset"] = [
        column in (cross_asset_feature_names or set())
        for column in score_frame.index
    ]
    score_frame["passes_activation_rate"] = (
        (score_frame["train_activation_rate"] >= selection_config.min_activation_rate)
        & (score_frame["train_activation_rate"] <= selection_config.max_activation_rate)
    )
    return score_frame.sort_values(
        ["passes_activation_rate", "mutual_info_mean", "train_activation_rate"],
        ascending=[False, False, False],
    )


def select_event_features(
    event_panel: pd.DataFrame,
    market_data: pd.DataFrame,
    training_config: TrainingConfig,
    selection_config: EventSelectionConfig,
    cross_asset_feature_names: set[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    feature_scores = score_event_features(
        event_panel=event_panel,
        market_data=market_data,
        training_config=training_config,
        selection_config=selection_config,
        cross_asset_feature_names=cross_asset_feature_names,
    )

    keep_mask = feature_scores["passes_activation_rate"].copy()
    if selection_config.min_mutual_info is not None:
        keep_mask &= feature_scores["mutual_info_mean"] >= float(selection_config.min_mutual_info)

    keep_features = feature_scores.index[keep_mask].tolist()

    if selection_config.preserve_cross_asset_events:
        cross_asset_keep = feature_scores.index[feature_scores["is_cross_asset"]].tolist()
        keep_features = list(dict.fromkeys(keep_features + cross_asset_keep))

    if selection_config.max_features is not None and len(keep_features) > selection_config.max_features:
        preserved = []
        if selection_config.preserve_cross_asset_events:
            preserved = [name for name in keep_features if bool(feature_scores.loc[name, "is_cross_asset"])]
        remaining_slots = max(0, int(selection_config.max_features) - len(preserved))
        ranked_non_preserved = [
            name
            for name in feature_scores.index.tolist()
            if name in keep_features and name not in preserved
        ]
        keep_features = preserved + ranked_non_preserved[:remaining_slots]

    keep_features = [name for name in event_panel.columns if name in set(keep_features)]
    if not keep_features:
        raise ValueError("Feature selection removed all event channels.")

    filtered_panel = event_panel.loc[:, keep_features].copy()
    dropped_features = [name for name in event_panel.columns if name not in keep_features]
    selection_metadata = {
        "feature_scores": feature_scores,
        "selected_features": keep_features,
        "dropped_features": dropped_features,
        "selection_applied": True,
        "selected_event_density": float(filtered_panel.to_numpy(dtype=np.float32).mean()),
    }
    return filtered_panel, selection_metadata


def prepare_event_experiment_inputs(
    data_config: SequenceDataConfig,
    event_config: EventEncodingConfig,
    training_config: TrainingConfig,
    selection_config: EventSelectionConfig | None = None,
) -> dict:
    event_panel, sequence_metadata, market_data, macro_data = build_event_panel(
        data_config=data_config,
        event_config=event_config,
    )
    cross_asset_feature_names = {
        column
        for column in event_panel.columns
        if not any(column.startswith(f"{asset}_") for asset in data_config.tuple_asset_columns)
    }
    feature_scores = None
    if selection_config is not None and selection_config.enabled:
        event_panel, selection_metadata = select_event_features(
            event_panel=event_panel,
            market_data=market_data,
            training_config=training_config,
            selection_config=selection_config,
            cross_asset_feature_names=cross_asset_feature_names,
        )
        sequence_metadata["used_features"] = selection_metadata["selected_features"]
        sequence_metadata["skipped_features"] = sequence_metadata["skipped_features"] + selection_metadata["dropped_features"]
        sequence_metadata["selection_applied"] = True
        sequence_metadata["selected_event_density"] = selection_metadata["selected_event_density"]
        feature_scores = selection_metadata["feature_scores"]

    row_split_indices = split_ordered_train_random_holdout_indices(
        n_obs=len(event_panel),
        train_ratio=training_config.train_ratio,
        validation_ratio=training_config.validation_ratio,
        test_ratio=training_config.test_ratio,
        random_state=training_config.random_state,
    )
    split_windows = build_split_windows(
        frame=event_panel,
        window_size=data_config.window_size,
        train_ratio=training_config.train_ratio,
        validation_ratio=training_config.validation_ratio,
        test_ratio=training_config.test_ratio,
        random_state=training_config.random_state,
    )
    return {
        "market_data": market_data,
        "macro_data": macro_data,
        "sequence_panel": event_panel,
        "sequence_metadata": sequence_metadata,
        "sequence_scaler": None,
        "windows": split_windows["windows"].astype(np.float32),
        "window_end_dates": split_windows["window_end_dates"],
        "splits": split_windows["splits"],
        "split_summary": split_windows["split_summary"],
        "window_split_indices": split_windows["window_split_indices"],
        "row_split_indices": row_split_indices,
        "feature_scores": feature_scores,
    }


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        return (input_tensor >= 0.0).to(input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (input_tensor,) = ctx.saved_tensors
        surrogate_grad = 1.0 / (1.0 + input_tensor.abs()).pow(2)
        return grad_output * surrogate_grad


def surrogate_spike(input_tensor: torch.Tensor) -> torch.Tensor:
    return _SurrogateSpike.apply(input_tensor)


class EventSpikeAutoencoder(nn.Module):
    def __init__(self, input_dim: int, model_config: EventSNNConfig):
        super().__init__()
        self.hidden_dim = int(model_config.hidden_dim)
        self.beta = float(model_config.beta)
        self.threshold = float(model_config.threshold)
        self.recurrent_scale = float(model_config.recurrent_scale)
        self.use_mask_embedding = bool(model_config.use_mask_embedding)

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.mask_proj = nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.recurrent = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.hidden_dropout = nn.Dropout(model_config.dropout)

        pooled_dim = self.hidden_dim * 2
        self.decode_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(self.hidden_dim, input_dim),
        )
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(self.hidden_dim, model_config.embedding_dim),
        )
        nn.init.xavier_uniform_(self.recurrent.weight)
        self.recurrent.weight.data.mul_(0.10)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, window_size, _ = x.shape
        normalized_x = self.input_norm(x)

        mem = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        prev_spike = torch.zeros_like(mem)
        spike_states = []
        membrane_states = []

        for step_idx in range(window_size):
            current = self.input_proj(normalized_x[:, step_idx, :])
            if mask is not None and self.use_mask_embedding:
                current = current + self.mask_proj(mask[:, step_idx, :].float())
            current = current + self.recurrent(prev_spike) * self.recurrent_scale
            mem = self.beta * mem + current
            spikes = surrogate_spike(mem - self.threshold)
            mem = mem - spikes * self.threshold
            spike_states.append(self.hidden_dropout(spikes))
            membrane_states.append(mem)
            prev_spike = spikes

        spike_trace = torch.stack(spike_states, dim=1)
        membrane_trace = torch.stack(membrane_states, dim=1)
        reconstruction = self.decode_head(torch.cat([spike_trace, membrane_trace], dim=-1))
        pooled = torch.cat([spike_trace.mean(dim=1), membrane_trace[:, -1, :]], dim=-1)
        embedding = self.embedding_head(pooled)
        return reconstruction, embedding


def sample_event_mask(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    mask = torch.rand_like(x) < mask_ratio
    flat_mask = mask.reshape(mask.size(0), -1)
    flat_events = (x > 0.5).reshape(x.size(0), -1)

    empty_rows = flat_mask.sum(dim=1) == 0
    if empty_rows.any():
        random_positions = torch.randint(0, flat_mask.size(1), (int(empty_rows.sum().item()),), device=x.device)
        flat_mask[empty_rows, random_positions] = True

    positive_rows = flat_events.sum(dim=1) > 0
    missed_positive_rows = positive_rows & ((flat_mask & flat_events).sum(dim=1) == 0)
    for row_idx in torch.nonzero(missed_positive_rows, as_tuple=False).flatten():
        positive_positions = torch.nonzero(flat_events[row_idx], as_tuple=False).flatten()
        choice = positive_positions[torch.randint(0, positive_positions.numel(), (1,), device=x.device)]
        flat_mask[row_idx, choice] = True
    return flat_mask.reshape_as(x)


def _compute_pos_weight(train_windows: torch.Tensor) -> torch.Tensor:
    positive_rate = train_windows.mean(dim=(0, 1)).clamp(min=1e-4, max=1.0 - 1e-4)
    pos_weight = ((1.0 - positive_rate) / positive_rate).clamp(min=1.0, max=25.0)
    return pos_weight


def masked_event_reconstruction_loss(
    model: nn.Module,
    batch: torch.Tensor,
    mask_ratio: float,
    pos_weight: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask is None:
        mask = sample_event_mask(batch, mask_ratio)
    masked_batch = batch.clone()
    masked_batch[mask] = 0.0
    reconstruction_logits, embeddings = model(masked_batch, mask=mask)

    loss_map = F.binary_cross_entropy_with_logits(
        reconstruction_logits,
        batch,
        reduction="none",
    )
    weighted_loss = torch.where(batch > 0.5, loss_map * pos_weight.view(1, 1, -1), loss_map)
    loss = weighted_loss[mask].mean()
    return loss, reconstruction_logits, embeddings


def train_event_snn(
    windows: np.ndarray,
    model_config: EventSNNConfig,
    training_config: TrainingConfig,
    split_indices: dict[str, np.ndarray],
) -> tuple[EventSpikeAutoencoder, pd.DataFrame, torch.Tensor, torch.device]:
    device = resolve_device(training_config.device)
    train_windows = torch.tensor(windows[split_indices["train"]], dtype=torch.float32)
    val_windows = torch.tensor(windows[split_indices["validation"]], dtype=torch.float32)
    all_windows = torch.tensor(windows, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_windows),
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=make_torch_dataloader_generator(training_config.random_state),
    )
    pos_weight = _compute_pos_weight(train_windows).to(device)

    if len(val_windows) > 0:
        val_masks = sample_event_mask(val_windows, training_config.mask_ratio)
        val_loader = DataLoader(
            TensorDataset(val_windows, val_masks),
            batch_size=training_config.batch_size,
            shuffle=False,
        )
    else:
        val_loader = None

    model = EventSpikeAutoencoder(
        input_dim=windows.shape[2],
        model_config=model_config,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history_rows = []
    best_state_dict = None
    best_val_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(1, training_config.num_epochs + 1):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, _, _ = masked_event_reconstruction_loss(
                model=model,
                batch=batch,
                mask_ratio=training_config.mask_ratio,
                pos_weight=pos_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch, val_mask in val_loader:
                    batch = batch.to(device)
                    val_mask = val_mask.to(device)
                    val_loss, _, _ = masked_event_reconstruction_loss(
                        model=model,
                        batch=batch,
                        mask_ratio=training_config.mask_ratio,
                        pos_weight=pos_weight,
                        mask=val_mask,
                    )
                    val_losses.append(float(val_loss.item()))
            val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        else:
            val_loss = train_loss

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if training_config.log_every and (epoch % training_config.log_every == 0 or epoch == training_config.num_epochs):
            print(
                f"[{model_config.architecture}] epoch={epoch}/{training_config.num_epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
                flush=True,
            )

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if (
            epoch < training_config.num_epochs
            and epoch >= training_config.min_epochs
            and no_improve_epochs >= training_config.early_stopping_patience
        ):
            print(f"[{model_config.architecture}] early stopping at epoch {epoch}", flush=True)
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, pd.DataFrame(history_rows), all_windows, device


def extract_event_snn_embeddings(
    model: EventSpikeAutoencoder,
    all_windows: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(all_windows), batch_size=128, shuffle=False)
    embeddings = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            _, embedding = model(batch)
            embeddings.append(embedding.cpu().numpy())

    return np.vstack(embeddings)


def run_event_snn_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    event_config: EventEncodingConfig | None = None,
    selection_config: EventSelectionConfig | None = None,
    model_config: EventSNNConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    data_config = data_config or build_default_track_c_data_config(TRACK_C_DIR.parent / "data")
    event_config = event_config or EventEncodingConfig()
    model_config = model_config or build_default_event_model_config()
    training_config = training_config or build_default_event_training_config()
    clustering_config = clustering_config or ClusteringConfig(target_cluster_count=4, cluster_candidates=(4,))
    hmm_config = hmm_config or HMMReferenceConfig(enabled=False, state_candidates=(4,))

    set_seed(training_config.random_state)
    prepared_inputs = prepare_event_experiment_inputs(
        data_config=data_config,
        event_config=event_config,
        training_config=training_config,
        selection_config=selection_config,
    )

    hmm_results = prepare_hmm_reference_for_experiment(
        data_config=data_config,
        training_config=training_config,
        hmm_config=hmm_config,
        market_data=prepared_inputs["market_data"],
        macro_data=prepared_inputs["macro_data"],
    )

    model, history_df, all_windows, device = train_event_snn(
        windows=prepared_inputs["windows"],
        model_config=model_config,
        training_config=training_config,
        split_indices=prepared_inputs["window_split_indices"],
    )
    embeddings = extract_event_snn_embeddings(model, all_windows, device)

    target_cluster_count = clustering_config.target_cluster_count
    if target_cluster_count is None and hmm_results is not None:
        target_cluster_count = int(hmm_results["best_n_states"])

    cluster_scan, cluster_model, cluster_labels, target_cluster_count = cluster_embeddings(
        embeddings=embeddings,
        clustering_config=clustering_config,
        random_state=training_config.random_state,
        target_cluster_count=target_cluster_count,
    )

    summary, cluster_summary, comparison_table = summarize_experiment_outputs(
        experiment_name=experiment_name,
        architecture=model_config.architecture,
        sequence_panel=prepared_inputs["sequence_panel"],
        sequence_metadata=prepared_inputs["sequence_metadata"],
        window_size=data_config.window_size,
        window_end_dates=prepared_inputs["window_end_dates"],
        history_df=history_df,
        cluster_scan=cluster_scan,
        cluster_labels=cluster_labels,
        target_cluster_count=target_cluster_count,
        hmm_results=hmm_results,
    )

    result = build_experiment_result(
        experiment_name=experiment_name,
        summary=summary,
        data_config=data_config,
        model_config={
            "event_config": event_config,
            "snn_config": model_config,
        },
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
        prepared_inputs=prepared_inputs,
        model=model,
        history_df=history_df,
        embeddings=embeddings,
        cluster_scan=cluster_scan,
        cluster_model=cluster_model,
        cluster_labels=cluster_labels,
        cluster_summary=cluster_summary,
        comparison_table=comparison_table,
        hmm_results=hmm_results,
        device=str(device),
    )
    result["event_config"] = event_config
    result["selection_config"] = selection_config
    result["snn_config"] = model_config
    result["feature_scores"] = prepared_inputs.get("feature_scores")
    return result


def save_event_snn_experiment(
    experiment_result: dict,
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "embeddings.npy", experiment_result["embeddings"])
    np.save(output_path / "cluster_labels.npy", experiment_result["cluster_labels"])

    pd.Index(experiment_result["window_end_dates"], name="Date").to_series().to_csv(
        output_path / "window_end_dates.csv",
        index=False,
    )
    experiment_result["history_df"].to_csv(output_path / "history_df.csv", index=False)
    experiment_result["cluster_scan"].to_csv(output_path / "cluster_scan.csv", index=False)
    experiment_result["cluster_summary"].to_csv(output_path / "cluster_summary.csv", index=False)
    if experiment_result.get("comparison_table") is not None:
        experiment_result["comparison_table"].to_csv(output_path / "comparison_table.csv")
    feature_scores = experiment_result.get("feature_scores")
    if isinstance(feature_scores, pd.DataFrame) and not feature_scores.empty:
        feature_scores.to_csv(output_path / "feature_scores.csv")

    model = experiment_result.get("model")
    model_path = None
    if isinstance(model, nn.Module):
        model_path = output_path / "model_state.pt"
        torch.save(model.state_dict(), model_path)

    metadata = {
        "experiment_name": experiment_result["experiment_name"],
        "summary": experiment_result["summary"],
        "event_config": asdict(experiment_result["event_config"]),
        "selection_config": (
            asdict(experiment_result["selection_config"])
            if experiment_result.get("selection_config") is not None
            else None
        ),
        "snn_config": asdict(experiment_result["snn_config"]),
        "training_config": asdict(experiment_result["training_config"]),
        "clustering_config": asdict(experiment_result["clustering_config"]),
        "hmm_config": asdict(experiment_result["hmm_config"]),
        "device": experiment_result.get("device"),
        "model_state_path": str(model_path) if model_path is not None else None,
        "sequence_metadata": experiment_result.get("sequence_metadata"),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "EventEncodingConfig",
    "EventSelectionConfig",
    "EventSNNConfig",
    "EventSpikeAutoencoder",
    "build_default_track_c_data_config",
    "build_default_event_training_config",
    "build_default_event_model_config",
    "build_event_panel",
    "score_event_features",
    "select_event_features",
    "prepare_event_experiment_inputs",
    "run_event_snn_experiment",
    "save_event_snn_experiment",
]
