from __future__ import annotations

import os
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from hmmlearn.hmm import GaussianHMM

    HMMLEARN_AVAILABLE = True
    HMMLEARN_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on local environment
    GaussianHMM = None
    HMMLEARN_AVAILABLE = False
    HMMLEARN_IMPORT_ERROR = exc


warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="MiniBatchKMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
    category=UserWarning,
)
torch.set_num_threads(1)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"


@dataclass
class SequenceDataConfig:
    data_dir: str | Path = DEFAULT_DATA_DIR
    market_filename: str = "market_data.csv"
    macro_filename: str = "macro_data.csv"
    hmm_features_filename: str = "hmm_features.csv"
    window_size: int = 60
    asset_return_columns: tuple[str, ...] = ("SPY", "TLT", "GLD", "UUP", "HYG", "LQD")
    include_vix_log_return: bool = True
    include_curve_slope_change: bool = True
    strict_feature_availability: bool = False
    tuple_market_value_field: str = "Close"
    sequence_feature_mode: str = "returns"
    tuple_asset_columns: tuple[str, ...] = ("GLD", "HYG", "LQD", "SPY", "TLT", "UUP", "VIX")
    tuple_market_fields: tuple[str, ...] = ("Close", "High", "Low", "Open", "Volume")
    tuple_log1p_volume: bool = True
    prefer_saved_hmm_features: bool = True


@dataclass
class EncoderOnlyTransformerConfig:
    d_model: int = 64
    embedding_dim: int = 64
    n_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.10
    feedforward_multiplier: float = 2.0
    pooling: str = "attention"
    use_mask_embedding: bool = True


@dataclass
class TrainingConfig:
    mask_ratio: float = 0.20
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    train_ratio: float = 0.7
    validation_ratio: float = 0.2
    test_ratio: float = 0.1
    early_stopping_patience: int = 5
    min_epochs: int = 5
    random_state: int = 42
    log_every: int = 1
    device: str = "auto"


@dataclass
class ClusteringConfig:
    target_cluster_count: int | None = None
    cluster_candidates: tuple[int, ...] = (2, 3, 4, 5, 6)
    n_init: int = 20
    batch_size: int = 256


@dataclass
class HMMReferenceConfig:
    enabled: bool = True
    state_candidates: tuple[int, ...] = (2, 3, 4)
    covariance_type: str = "diag"
    n_restarts: int = 6
    n_iter: int = 500


def _resolve_data_dir(data_dir: str | Path) -> Path:
    return Path(data_dir).resolve()


def _coerce_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_frame = frame.copy()
    for column in numeric_frame.columns:
        numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")
    return numeric_frame


def _extract_tuple_field_value(value: object, position: int) -> float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    if not (text.startswith("(") and text.endswith(")")):
        try:
            return float(text)
        except ValueError:
            return np.nan

    parts = [part.strip() for part in text[1:-1].split(",")]
    if position >= len(parts):
        return np.nan

    token = parts[position]
    if not token or token.lower() == "null":
        return np.nan

    try:
        return float(token)
    except ValueError:
        return np.nan


def _coerce_market_frame(
    market_data: pd.DataFrame,
    value_field: str,
) -> pd.DataFrame:
    field_positions = {"Close": 0, "High": 1, "Low": 2, "Open": 3, "Volume": 4}
    if value_field not in field_positions:
        raise ValueError(
            f"Unsupported tuple_market_value_field `{value_field}`. "
            f"Choose one of: {', '.join(field_positions)}."
        )

    position = field_positions[value_field]
    numeric_market = market_data.copy()

    for column in numeric_market.columns:
        series = numeric_market[column]
        sample = series.dropna().head(1)
        if sample.empty:
            numeric_market[column] = pd.to_numeric(series, errors="coerce")
            continue

        first_value = sample.iloc[0]
        if isinstance(first_value, str) and first_value.strip().startswith("("):
            numeric_market[column] = series.map(lambda value: _extract_tuple_field_value(value, position))
        else:
            numeric_market[column] = pd.to_numeric(series, errors="coerce")

    if "^VIX" in numeric_market.columns and "VIX" not in numeric_market.columns:
        numeric_market = numeric_market.rename(columns={"^VIX": "VIX"})

    return numeric_market


def _load_raw_market_data(data_config: SequenceDataConfig) -> pd.DataFrame:
    data_dir = _resolve_data_dir(data_config.data_dir)
    market_path = data_dir / data_config.market_filename
    market_data = pd.read_csv(market_path, index_col="Date", parse_dates=True).sort_index()
    market_data.index.name = "Date"
    return market_data


def load_market_tuple_panels(
    data_config: SequenceDataConfig,
    target_index: pd.Index | None = None,
) -> dict[str, pd.DataFrame]:
    raw_market_data = _load_raw_market_data(data_config)
    field_panels: dict[str, pd.DataFrame] = {}
    for field_name in data_config.tuple_market_fields:
        panel = _coerce_market_frame(raw_market_data, value_field=field_name)
        if target_index is not None:
            common_index = panel.index.intersection(target_index)
            panel = panel.loc[common_index].copy()
        panel.index.name = "Date"
        field_panels[field_name] = panel
    return field_panels


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def make_torch_dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def load_market_and_macro(data_config: SequenceDataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = _resolve_data_dir(data_config.data_dir)
    market_path = data_dir / data_config.market_filename
    macro_path = data_dir / data_config.macro_filename

    market_data = pd.read_csv(market_path, index_col="Date", parse_dates=True).sort_index()
    macro_data = pd.read_csv(macro_path, index_col="Date", parse_dates=True).sort_index()

    market_data = _coerce_market_frame(market_data, value_field=data_config.tuple_market_value_field)
    macro_data = _coerce_numeric_frame(macro_data)

    common_index = market_data.index.intersection(macro_data.index)
    market_data = market_data.loc[common_index].copy()
    macro_data = macro_data.loc[common_index].copy()

    market_data.index.name = "Date"
    macro_data.index.name = "Date"

    return market_data, macro_data


def build_sequence_panel(
    market_data: pd.DataFrame,
    macro_data: pd.DataFrame,
    data_config: SequenceDataConfig,
) -> tuple[pd.DataFrame, dict]:
    if data_config.sequence_feature_mode == "market_tuples":
        tuple_panels = load_market_tuple_panels(data_config, target_index=market_data.index)
        sequence_features: list[pd.Series] = []
        used_features: list[str] = []
        skipped_features: list[str] = []

        available_assets = list(tuple_panels[next(iter(tuple_panels))].columns) if tuple_panels else []
        tuple_assets = list(data_config.tuple_asset_columns) if data_config.tuple_asset_columns else available_assets

        for asset in tuple_assets:
            for field_name in data_config.tuple_market_fields:
                field_panel = tuple_panels.get(field_name)
                if field_panel is None or asset not in field_panel.columns:
                    skipped_features.append(f"{asset}_{field_name}")
                    continue

                series = field_panel[asset].copy()
                if not series.notna().any():
                    skipped_features.append(f"{asset}_{field_name}")
                    continue

                feature_name = f"{asset}_{field_name.lower()}"
                if field_name == "Volume" and data_config.tuple_log1p_volume:
                    series = np.log1p(series.clip(lower=0))
                    feature_name = f"{asset}_log1p_volume"

                sequence_features.append(series.rename(feature_name))
                used_features.append(feature_name)

        if data_config.strict_feature_availability and skipped_features:
            missing_text = ", ".join(skipped_features)
            raise ValueError(f"Missing required tuple market features: {missing_text}")

        if not sequence_features:
            raise ValueError("No usable tuple-based market features were built from the full market data file.")

        sequence_panel = pd.concat(sequence_features, axis=1).dropna()
        sequence_panel.index.name = "Date"

        if sequence_panel.empty:
            raise ValueError(
                "Tuple-based sequence panel is empty after dropna(). "
                "Check whether the requested OHLCV tuple fields are available."
            )

        metadata = {
            "used_features": used_features,
            "skipped_features": skipped_features,
        }
        return sequence_panel, metadata

    if data_config.sequence_feature_mode != "returns":
        raise ValueError(
            f"Unsupported sequence_feature_mode `{data_config.sequence_feature_mode}`. "
            "Choose `returns` or `market_tuples`."
        )

    sequence_features: list[pd.Series] = []
    used_features: list[str] = []
    skipped_features: list[str] = []

    returns = market_data.pct_change(fill_method=None)

    for asset in data_config.asset_return_columns:
        feature_name = f"{asset}_ret"
        if asset in returns.columns and returns[asset].notna().any():
            sequence_features.append(returns[asset].rename(feature_name))
            used_features.append(feature_name)
        else:
            skipped_features.append(feature_name)

    if data_config.include_vix_log_return:
        if "VIX" in market_data.columns and market_data["VIX"].notna().any():
            vix_ret = np.log(market_data["VIX"]).diff().rename("VIX_ret")
            sequence_features.append(vix_ret)
            used_features.append("VIX_ret")
        else:
            skipped_features.append("VIX_ret")

    if data_config.include_curve_slope_change:
        if {"DGS10", "DGS2"}.issubset(macro_data.columns):
            curve_slope_change = (macro_data["DGS10"] - macro_data["DGS2"]).diff().rename("curve_slope_change")
            sequence_features.append(curve_slope_change)
            used_features.append("curve_slope_change")
        else:
            skipped_features.append("curve_slope_change")

    if data_config.strict_feature_availability and skipped_features:
        missing_text = ", ".join(skipped_features)
        raise ValueError(f"Missing required Track B input features: {missing_text}")

    if not sequence_features:
        raise ValueError("No usable Track B sequence features were built from market_data.csv and macro_data.csv.")

    sequence_panel = pd.concat(sequence_features, axis=1).dropna()
    sequence_panel.index.name = "Date"

    if sequence_panel.empty:
        raise ValueError(
            "Sequence panel is empty after dropna(). Check whether required market or macro columns are missing."
        )

    metadata = {
        "used_features": used_features,
        "skipped_features": skipped_features,
    }
    return sequence_panel, metadata


def make_windows(frame: pd.DataFrame, window_size: int) -> tuple[np.ndarray, pd.Index]:
    if len(frame) < window_size:
        raise ValueError(f"Not enough rows to build windows: len(frame)={len(frame)}, window_size={window_size}")

    values = frame.values.astype(np.float32)
    windows = []
    end_dates = []

    for end in range(window_size, len(frame) + 1):
        start = end - window_size
        windows.append(values[start:end])
        end_dates.append(frame.index[end - 1])

    return np.stack(windows), pd.Index(end_dates, name="Date")


def summarize_date_splits(splits: dict[str, pd.Index]) -> pd.DataFrame:
    rows = []
    for split_name in ("train", "validation", "test"):
        split_dates = pd.Index(splits.get(split_name, pd.Index([], name="Date")), name="Date")
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


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class EncoderOnlyMaskedTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_config: EncoderOnlyTransformerConfig,
        max_len: int,
    ):
        super().__init__()
        if model_config.pooling not in {"attention", "mean", "last", "cls"}:
            raise ValueError("pooling must be one of: attention, mean, last, cls")
        if model_config.d_model % model_config.n_heads != 0:
            raise ValueError(
                "Invalid encoder_only configuration: "
                f"d_model={model_config.d_model} must be divisible by "
                f"n_heads={model_config.n_heads}."
            )

        feedforward_dim = int(model_config.d_model * model_config.feedforward_multiplier)
        self.pooling = model_config.pooling
        self.use_mask_embedding = model_config.use_mask_embedding
        self.use_cls_token = model_config.pooling == "cls"

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, model_config.d_model)
        self.mask_proj = nn.Linear(input_dim, model_config.d_model, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_config.d_model)) if self.use_cls_token else None
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=model_config.d_model,
            max_len=max_len + (1 if self.use_cls_token else 0),
        )
        self.pos_dropout = nn.Dropout(model_config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config.d_model,
            nhead=model_config.n_heads,
            dim_feedforward=feedforward_dim,
            dropout=model_config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config.num_layers,
            norm=nn.LayerNorm(model_config.d_model),
        )

        self.reconstruct_head = nn.Sequential(
            nn.LayerNorm(model_config.d_model),
            nn.Linear(model_config.d_model, model_config.d_model),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.d_model, input_dim),
        )

        self.pool_score = nn.Linear(model_config.d_model, 1) if self.pooling == "attention" else None
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(model_config.d_model),
            nn.Linear(model_config.d_model, model_config.d_model),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.d_model, model_config.embedding_dim),
        )

    def pool_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0, :]
        if self.pooling == "attention":
            assert self.pool_score is not None
            weights = torch.softmax(self.pool_score(hidden), dim=1)
            return (hidden * weights).sum(dim=1)
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        return hidden[:, -1, :]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_x = self.input_norm(x)
        hidden = self.input_proj(normalized_x)
        if mask is not None and self.use_mask_embedding:
            hidden = hidden + self.mask_proj(mask.float())
        if self.use_cls_token:
            assert self.cls_token is not None
            cls_token = self.cls_token.expand(hidden.size(0), -1, -1)
            hidden = torch.cat([cls_token, hidden], dim=1)
        hidden = self.pos_dropout(self.pos_encoder(hidden))
        hidden = self.encoder(hidden)
        reconstruction_hidden = hidden[:, 1:, :] if self.use_cls_token else hidden
        reconstruction = self.reconstruct_head(reconstruction_hidden)
        pooled = self.pool_hidden(hidden)
        embedding = self.embedding_head(pooled)
        return reconstruction, embedding


def sample_mask(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    mask = torch.rand_like(x) < mask_ratio
    flat = mask.reshape(mask.size(0), -1)
    empty_rows = flat.sum(dim=1) == 0
    if empty_rows.any():
        random_positions = torch.randint(0, flat.size(1), (int(empty_rows.sum().item()),), device=x.device)
        flat[empty_rows, random_positions] = True
    return flat.reshape_as(mask)


def masked_reconstruction_loss(
    model: nn.Module,
    batch: torch.Tensor,
    mask_ratio: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask is None:
        mask = sample_mask(batch, mask_ratio)
    masked_batch = batch.clone()
    masked_batch[mask] = 0.0
    reconstruction, embeddings = model(masked_batch, mask=mask)
    loss = ((reconstruction - batch) ** 2)[mask].mean()
    return loss, reconstruction, embeddings


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def split_ordered_train_random_holdout_indices(
    n_obs: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_state: int,
) -> dict[str, np.ndarray]:
    if n_obs < 3:
        raise ValueError(f"Need at least 3 observations to split train/validation/test, got {n_obs}.")

    total_ratio = train_ratio + validation_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("train_ratio + validation_ratio + test_ratio must be positive.")

    normalized_train_ratio = train_ratio / total_ratio
    normalized_validation_ratio = validation_ratio / total_ratio
    normalized_test_ratio = test_ratio / total_ratio

    train_count = max(1, int(n_obs * normalized_train_ratio))
    train_count = min(train_count, n_obs - 2)
    holdout_indices = np.arange(train_count, n_obs, dtype=int)
    if len(holdout_indices) < 2:
        raise ValueError(
            "Need at least 2 holdout observations after the training split to form validation/test sets."
        )

    holdout_total_ratio = normalized_validation_ratio + normalized_test_ratio
    if holdout_total_ratio <= 0:
        raise ValueError("validation_ratio + test_ratio must be positive.")

    validation_share = normalized_validation_ratio / holdout_total_ratio
    validation_count = int(round(len(holdout_indices) * validation_share))
    validation_count = max(1, validation_count)
    validation_count = min(validation_count, len(holdout_indices) - 1)

    validation_indices = holdout_indices[:validation_count]
    test_indices = holdout_indices[validation_count:]

    return {
        "train": np.arange(train_count, dtype=int),
        "validation": validation_indices,
        "test": test_indices,
    }


def build_split_windows(
    frame: pd.DataFrame,
    window_size: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_state: int,
) -> dict:
    row_split_indices = split_ordered_train_random_holdout_indices(
        n_obs=len(frame),
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    windows_by_split: dict[str, np.ndarray] = {}
    split_dates: dict[str, pd.Index] = {}
    window_split_indices: dict[str, np.ndarray] = {}
    ordered_windows: list[np.ndarray] = []
    ordered_dates: list[pd.Index] = []
    cursor = 0

    for split_name in ("train", "validation", "test"):
        row_indices = np.asarray(row_split_indices[split_name], dtype=int)
        split_frame = frame.iloc[row_indices].copy()
        if len(split_frame) < window_size:
            raise ValueError(
                f"{split_name} split has only {len(split_frame)} rows, which is not enough "
                f"to form a {window_size}-day window."
            )

        split_windows, split_window_end_dates = make_windows(split_frame, window_size)
        windows_by_split[split_name] = split_windows
        split_dates[split_name] = pd.Index(split_window_end_dates, name="Date")
        ordered_windows.append(split_windows)
        ordered_dates.append(pd.Index(split_window_end_dates, name="Date"))

        split_window_count = len(split_windows)
        window_split_indices[split_name] = np.arange(cursor, cursor + split_window_count, dtype=int)
        cursor += split_window_count

    windows = np.concatenate(ordered_windows, axis=0)
    window_end_dates = pd.Index(np.concatenate([dates.to_numpy() for dates in ordered_dates]), name="Date")

    return {
        "windows": windows,
        "window_end_dates": window_end_dates,
        "splits": split_dates,
        "split_summary": summarize_date_splits(split_dates),
        "window_split_indices": window_split_indices,
        "row_split_indices": row_split_indices,
    }


def train_encoder_only_transformer(
    windows: np.ndarray,
    model_config: EncoderOnlyTransformerConfig,
    training_config: TrainingConfig,
    window_size: int,
    split_indices: dict[str, np.ndarray] | None = None,
) -> tuple[EncoderOnlyMaskedTransformer, pd.DataFrame, torch.Tensor, torch.device]:
    device = resolve_device(training_config.device)
    if split_indices is None:
        split_indices = split_ordered_train_random_holdout_indices(
            n_obs=len(windows),
            train_ratio=training_config.train_ratio,
            validation_ratio=training_config.validation_ratio,
            test_ratio=training_config.test_ratio,
            random_state=training_config.random_state,
        )

    train_windows = torch.tensor(windows[split_indices["train"]], dtype=torch.float32)
    val_windows = torch.tensor(windows[split_indices["validation"]], dtype=torch.float32)
    all_windows = torch.tensor(windows, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_windows),
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=make_torch_dataloader_generator(training_config.random_state),
    )
    if len(val_windows) > 0:
        val_masks = sample_mask(val_windows, training_config.mask_ratio)
        val_loader = DataLoader(
            TensorDataset(val_windows, val_masks),
            batch_size=training_config.batch_size,
            shuffle=False,
        )
    else:
        val_loader = None

    model = EncoderOnlyMaskedTransformer(
        input_dim=windows.shape[2],
        model_config=model_config,
        max_len=window_size,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history = []
    best_state_dict = None
    best_val_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(1, training_config.num_epochs + 1):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, _, _ = masked_reconstruction_loss(model, batch, training_config.mask_ratio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch, val_mask in val_loader:
                    batch = batch.to(device)
                    val_mask = val_mask.to(device)
                    val_loss, _, _ = masked_reconstruction_loss(
                        model,
                        batch,
                        training_config.mask_ratio,
                        mask=val_mask,
                    )
                    val_losses.append(float(val_loss.item()))
            val_loss_value = float(np.mean(val_losses))
        else:
            val_loss_value = train_loss

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss_value})

        if training_config.log_every and (epoch % training_config.log_every == 0 or epoch == training_config.num_epochs):
            print(
                f"[encoder_only] epoch={epoch}/{training_config.num_epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss_value:.6f}",
                flush=True,
            )

        if val_loss_value <= best_val_loss:
            best_val_loss = val_loss_value
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if (
            epoch < training_config.num_epochs
            and epoch >= training_config.min_epochs
            and no_improve_epochs >= training_config.early_stopping_patience
        ):
            print(f"[encoder_only] early stopping at epoch {epoch}", flush=True)
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, pd.DataFrame(history), all_windows, device


def extract_embeddings(
    model: EncoderOnlyMaskedTransformer,
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


def fit_best_hmm(
    X: np.ndarray,
    n_states: int,
    covariance_type: str,
    n_restarts: int,
    n_iter: int,
    random_state: int,
) -> tuple[GaussianHMM, float]:
    if GaussianHMM is None:
        raise ImportError(
            "hmmlearn is required for HMM reference fitting. "
            "Install it with `pip install hmmlearn` or disable HMMReferenceConfig(enabled=False)."
        ) from HMMLEARN_IMPORT_ERROR

    best_model = None
    best_score = -np.inf

    for seed in range(random_state, random_state + n_restarts):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=seed,
        )
        model.fit(X)
        score = model.score(X)
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError("Failed to fit any HMM model for the given configuration.")

    return best_model, best_score


def hmm_parameter_count(n_states: int, n_features: int, covariance_type: str) -> int:
    startprob = n_states - 1
    transmat = n_states * (n_states - 1)
    means = n_states * n_features

    if covariance_type == "diag":
        covars = n_states * n_features
    elif covariance_type == "full":
        covars = n_states * n_features * (n_features + 1) // 2
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    return startprob + transmat + means + covars


def select_hmm_by_bic(
    X: np.ndarray,
    hmm_config: HMMReferenceConfig,
    random_state: int,
) -> tuple[pd.DataFrame, dict[int, GaussianHMM], int]:
    rows = []
    models = {}
    n_features = X.shape[1]
    n_obs = X.shape[0]

    for n_states in hmm_config.state_candidates:
        model, loglik = fit_best_hmm(
            X=X,
            n_states=n_states,
            covariance_type=hmm_config.covariance_type,
            n_restarts=hmm_config.n_restarts,
            n_iter=hmm_config.n_iter,
            random_state=random_state,
        )
        n_params = hmm_parameter_count(n_states, n_features, covariance_type=hmm_config.covariance_type)
        bic = -2 * loglik + n_params * np.log(n_obs)
        rows.append({"n_states": n_states, "loglik": loglik, "n_params": n_params, "bic": bic})
        models[n_states] = model

    summary = pd.DataFrame(rows).sort_values("n_states").reset_index(drop=True)
    best_n_states = int(summary.sort_values("bic").iloc[0]["n_states"])
    return summary, models, best_n_states


def label_hmm_states(feature_frame: pd.DataFrame, state_col: str = "hmm_state") -> tuple[dict[int, str], list[int], pd.Series]:
    metric_candidates = [
        "SPY_ret",
        "SPY_ret_vol_21d",
        "SPY_vol_21d",
        "VIX_level",
        "credit_spread_proxy",
        "curve_slope_10y2y",
        "corr_60d_SPY_TLT",
    ]
    metric_cols = [column for column in metric_candidates if column in feature_frame.columns]
    state_means = feature_frame.groupby(state_col)[metric_cols].mean()
    state_stds = state_means.std(ddof=0).replace(0, 1)
    z_scores = (state_means - state_means.mean()) / state_stds

    risk_score = pd.Series(0.0, index=state_means.index)
    if "SPY_ret" in z_scores.columns:
        risk_score += z_scores["SPY_ret"].fillna(0)
    if "SPY_ret_vol_21d" in z_scores.columns:
        risk_score -= z_scores["SPY_ret_vol_21d"].fillna(0)
    if "SPY_vol_21d" in z_scores.columns:
        risk_score -= z_scores["SPY_vol_21d"].fillna(0)
    if "VIX_level" in z_scores.columns:
        risk_score -= z_scores["VIX_level"].fillna(0)
    if "credit_spread_proxy" in z_scores.columns:
        risk_score += z_scores["credit_spread_proxy"].fillna(0)
    if "curve_slope_10y2y" in z_scores.columns:
        risk_score += 0.75 * z_scores["curve_slope_10y2y"].fillna(0)
    if "corr_60d_SPY_TLT" in z_scores.columns:
        risk_score -= 0.5 * z_scores["corr_60d_SPY_TLT"].fillna(0)

    ordered_states = risk_score.sort_values().index.tolist()
    n_states = len(ordered_states)

    if n_states == 2:
        names = ["risk-off", "risk-on"]
    elif n_states == 3:
        names = ["stress", "transition", "risk-on"]
    elif n_states == 4:
        names = ["stress", "transition", "recovery", "risk-on"]
    else:
        names = [f"state_{idx}" for idx in range(n_states)]

    label_map = {state: names[idx] for idx, state in enumerate(ordered_states)}
    return label_map, ordered_states, risk_score.sort_values()


def load_hmm_features(
    data_config: SequenceDataConfig,
    market_data: pd.DataFrame,
    macro_data: pd.DataFrame,
) -> pd.DataFrame:
    data_dir = _resolve_data_dir(data_config.data_dir)
    hmm_path = data_dir / data_config.hmm_features_filename
    use_saved_hmm_features = (
        data_config.prefer_saved_hmm_features
        and data_config.market_filename == "market_data.csv"
        and hmm_path.exists()
    )

    if use_saved_hmm_features:
        hmm_features = pd.read_csv(hmm_path, index_col="Date", parse_dates=True).sort_index()
        hmm_features.index.name = "Date"
        return hmm_features

    returns = market_data.pct_change(fill_method=None)
    fallback_features = pd.DataFrame(
        {
            "SPY_ret": returns["SPY"],
            "SPY_vol_21d": returns["SPY"].rolling(21).std() * np.sqrt(252),
            "VIX_level": market_data["VIX"],
            "TLT_ret": returns["TLT"],
            "GLD_ret": returns["GLD"],
            "UUP_ret": returns["UUP"],
            "curve_slope_10y2y": macro_data["DGS10"] - macro_data["DGS2"],
            "corr_60d_SPY_TLT": returns["SPY"].rolling(60).corr(returns["TLT"]),
        }
    )
    fallback_features.index.name = "Date"
    return fallback_features.dropna()


def build_hmm_reference(
    data_config: SequenceDataConfig,
    market_data: pd.DataFrame,
    macro_data: pd.DataFrame,
    hmm_config: HMMReferenceConfig,
    random_state: int,
) -> dict:
    if not HMMLEARN_AVAILABLE:
        raise ImportError(
            "hmmlearn is not installed, so HMM reference comparison is unavailable. "
            "Install it with `pip install hmmlearn` or run with HMMReferenceConfig(enabled=False)."
        ) from HMMLEARN_IMPORT_ERROR

    hmm_features = load_hmm_features(data_config, market_data, macro_data)
    scaler = StandardScaler()
    hmm_matrix = scaler.fit_transform(hmm_features)

    bic_summary, model_store, best_n_states = select_hmm_by_bic(
        hmm_matrix,
        hmm_config=hmm_config,
        random_state=random_state,
    )
    best_model = model_store[best_n_states]
    hmm_states = pd.Series(best_model.predict(hmm_matrix), index=hmm_features.index, name="hmm_state")

    hmm_reference = hmm_features.copy()
    hmm_reference["hmm_state"] = hmm_states
    label_map, ordered_states, risk_score = label_hmm_states(hmm_reference, state_col="hmm_state")
    hmm_reference["hmm_regime"] = hmm_reference["hmm_state"].map(label_map)

    return {
        "hmm_features": hmm_features,
        "hmm_scaler": scaler,
        "bic_summary": bic_summary,
        "best_n_states": best_n_states,
        "hmm_model": best_model,
        "hmm_reference": hmm_reference,
        "label_map": label_map,
        "ordered_states": ordered_states,
        "risk_score": risk_score,
    }


def cluster_embeddings(
    embeddings: np.ndarray,
    clustering_config: ClusteringConfig,
    random_state: int,
    target_cluster_count: int | None = None,
) -> tuple[pd.DataFrame, MiniBatchKMeans, np.ndarray, int]:
    if len(embeddings) < 3:
        raise ValueError("Need at least 3 windows to run Track B clustering.")

    cluster_scan_rows = []
    valid_candidates = [k for k in clustering_config.cluster_candidates if 1 < k < len(embeddings)]
    for n_clusters in valid_candidates:
        scan_model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=min(10, clustering_config.n_init),
            batch_size=clustering_config.batch_size,
        )
        labels = scan_model.fit_predict(embeddings)
        silhouette = silhouette_score(
            embeddings,
            labels,
            sample_size=min(1500, len(embeddings)),
            random_state=random_state,
        )
        cluster_scan_rows.append({"n_clusters": n_clusters, "silhouette": silhouette})

    cluster_scan = pd.DataFrame(cluster_scan_rows)

    if target_cluster_count is None:
        if cluster_scan.empty:
            target_cluster_count = 2
        else:
            target_cluster_count = int(cluster_scan.sort_values("silhouette", ascending=False).iloc[0]["n_clusters"])

    cluster_model = MiniBatchKMeans(
        n_clusters=target_cluster_count,
        random_state=random_state,
        n_init=clustering_config.n_init,
        batch_size=clustering_config.batch_size,
    )
    cluster_labels = cluster_model.fit_predict(embeddings)

    return cluster_scan, cluster_model, cluster_labels, target_cluster_count


def summarize_cluster_labels(cluster_labels: np.ndarray, end_dates: pd.Index) -> pd.DataFrame:
    return (
        pd.DataFrame({"Date": end_dates, "cluster": cluster_labels})
        .groupby("cluster")
        .agg(n_windows=("cluster", "size"))
        .reset_index()
        .sort_values("cluster")
        .reset_index(drop=True)
    )


def run_encoder_only_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: EncoderOnlyTransformerConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
    architecture_name: str = "encoder_only",
) -> dict:
    data_config = data_config or SequenceDataConfig()
    model_config = model_config or EncoderOnlyTransformerConfig()
    training_config = training_config or TrainingConfig()
    clustering_config = clustering_config or ClusteringConfig()
    hmm_config = hmm_config or HMMReferenceConfig()

    set_seed(training_config.random_state)

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
    prepared_windows = build_split_windows(
        frame=sequence_z,
        window_size=data_config.window_size,
        train_ratio=training_config.train_ratio,
        validation_ratio=training_config.validation_ratio,
        test_ratio=training_config.test_ratio,
        random_state=training_config.random_state,
    )
    windows = prepared_windows["windows"]
    window_end_dates = prepared_windows["window_end_dates"]
    splits = prepared_windows["splits"]
    split_summary = prepared_windows["split_summary"]
    window_split_indices = prepared_windows["window_split_indices"]

    hmm_results = None
    target_cluster_count = clustering_config.target_cluster_count
    if hmm_config.enabled:
        hmm_results = build_hmm_reference(
            data_config=data_config,
            market_data=market_data,
            macro_data=macro_data,
            hmm_config=hmm_config,
            random_state=training_config.random_state,
        )
        if target_cluster_count is None:
            target_cluster_count = int(hmm_results["best_n_states"])

    model, history_df, all_windows, device = train_encoder_only_transformer(
        windows=windows,
        model_config=model_config,
        training_config=training_config,
        window_size=data_config.window_size,
        split_indices=window_split_indices,
    )
    embeddings = extract_embeddings(model, all_windows, device)
    cluster_scan, cluster_model, cluster_labels, target_cluster_count = cluster_embeddings(
        embeddings=embeddings,
        clustering_config=clustering_config,
        random_state=training_config.random_state,
        target_cluster_count=target_cluster_count,
    )

    silhouette_value = np.nan
    if not cluster_scan.empty:
        matched = cluster_scan.loc[cluster_scan["n_clusters"] == target_cluster_count, "silhouette"]
        if not matched.empty:
            silhouette_value = float(matched.iloc[0])

    comparison_table = None
    cluster_summary = summarize_cluster_labels(cluster_labels, window_end_dates)
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
            if "curve_slope_10y2y" in cluster_profile.columns:
                agg_spec["avg_curve_slope_10y2y"] = ("curve_slope_10y2y", "mean")

            summary_rows = cluster_profile.groupby("cluster").agg(**agg_spec).reset_index()
            cluster_summary = summary_rows

    best_epoch = int(history_df.loc[history_df["val_loss"].idxmin(), "epoch"])
    best_val_loss = float(history_df["val_loss"].min())
    summary = {
        "experiment_name": experiment_name,
        "architecture": architecture_name,
        "input_dim": int(sequence_panel.shape[1]),
        "window_size": int(data_config.window_size),
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

    return {
        "experiment_name": experiment_name,
        "summary": summary,
        "data_config": data_config,
        "model_config": model_config,
        "training_config": training_config,
        "clustering_config": clustering_config,
        "hmm_config": hmm_config,
        "market_data": market_data,
        "macro_data": macro_data,
        "sequence_panel": sequence_panel,
        "sequence_metadata": sequence_metadata,
        "sequence_scaler": sequence_scaler,
        "windows": windows,
        "window_end_dates": window_end_dates,
        "splits": splits,
        "split_summary": split_summary,
        "window_split_indices": window_split_indices,
        "model": model,
        "history_df": history_df,
        "embeddings": embeddings,
        "cluster_scan": cluster_scan,
        "cluster_model": cluster_model,
        "cluster_labels": cluster_labels,
        "cluster_summary": cluster_summary,
        "comparison_table": comparison_table,
        "hmm_results": hmm_results,
        "device": str(device),
    }


def run_market_tuple_encoder_only_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: EncoderOnlyTransformerConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    if data_config is None:
        data_config = SequenceDataConfig(
            market_filename="market_data_full_adjusted.csv",
            window_size=60,
            sequence_feature_mode="market_tuples",
            include_vix_log_return=False,
            include_curve_slope_change=False,
        )

    return run_encoder_only_experiment(
        experiment_name=experiment_name,
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
        architecture_name="encoder_only_market_tuples",
    )


def compare_experiment_summaries(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        if "summary" in result:
            rows.append(result["summary"])
        else:
            rows.append(result)
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df
    sort_columns = [column for column in ["ari_vs_hmm", "nmi_vs_hmm", "silhouette"] if column in summary_df.columns]
    if sort_columns:
        summary_df = summary_df.sort_values(sort_columns, ascending=False, na_position="last")
    return summary_df.reset_index(drop=True)


__all__ = [
    "HMMLEARN_AVAILABLE",
    "SequenceDataConfig",
    "EncoderOnlyTransformerConfig",
    "TrainingConfig",
    "ClusteringConfig",
    "HMMReferenceConfig",
    "EncoderOnlyMaskedTransformer",
    "build_sequence_panel",
    "make_windows",
    "build_split_windows",
    "summarize_date_splits",
    "split_ordered_train_random_holdout_indices",
    "run_encoder_only_experiment",
    "run_market_tuple_encoder_only_experiment",
    "compare_experiment_summaries",
]
