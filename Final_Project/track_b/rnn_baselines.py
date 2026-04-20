from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from encoder_only_transformer import (
    ClusteringConfig,
    HMMLEARN_AVAILABLE,
    HMMReferenceConfig,
    SequenceDataConfig,
    TrainingConfig,
    build_hmm_reference,
    build_sequence_panel,
    cluster_embeddings,
    load_market_and_macro,
    make_windows,
    masked_reconstruction_loss,
    make_torch_dataloader_generator,
    resolve_device,
    set_seed,
    split_ordered_train_random_holdout_indices,
    summarize_cluster_labels,
)


@dataclass
class RecurrentBaselineConfig:
    architecture: str = "pure_rnn"
    input_projection_dim: int = 32
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 2
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 1
    embedding_dim: int = 64
    dropout: float = 0.10
    pooling: str = "attention"
    nonlinearity: str = "tanh"
    bidirectional: bool = False
    use_mask_embedding: bool = True


class RecurrentMaskedAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_config: RecurrentBaselineConfig,
    ):
        super().__init__()
        if model_config.architecture not in {"pure_rnn", "rnn_lstm_hybrid"}:
            raise ValueError("architecture must be one of: pure_rnn, rnn_lstm_hybrid")
        if model_config.pooling not in {"attention", "mean", "last"}:
            raise ValueError("pooling must be one of: attention, mean, last")

        self.architecture = model_config.architecture
        self.pooling = model_config.pooling
        self.use_mask_embedding = model_config.use_mask_embedding
        self.bidirectional = model_config.bidirectional

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, model_config.input_projection_dim)
        self.mask_proj = nn.Linear(input_dim, model_config.input_projection_dim, bias=False)

        rnn_dropout = model_config.dropout if model_config.rnn_num_layers > 1 else 0.0
        self.rnn = nn.RNN(
            input_size=model_config.input_projection_dim,
            hidden_size=model_config.rnn_hidden_dim,
            num_layers=model_config.rnn_num_layers,
            nonlinearity=model_config.nonlinearity,
            dropout=rnn_dropout,
            batch_first=True,
            bidirectional=model_config.bidirectional,
        )

        recurrent_output_dim = model_config.rnn_hidden_dim * (2 if model_config.bidirectional else 1)

        if self.architecture == "rnn_lstm_hybrid":
            lstm_dropout = model_config.dropout if model_config.lstm_num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=recurrent_output_dim,
                hidden_size=model_config.lstm_hidden_dim,
                num_layers=model_config.lstm_num_layers,
                dropout=lstm_dropout,
                batch_first=True,
                bidirectional=model_config.bidirectional,
            )
            recurrent_output_dim = model_config.lstm_hidden_dim * (2 if model_config.bidirectional else 1)
        else:
            self.lstm = None

        self.reconstruct_head = nn.Sequential(
            nn.LayerNorm(recurrent_output_dim),
            nn.Linear(recurrent_output_dim, recurrent_output_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(recurrent_output_dim, input_dim),
        )

        self.pool_score = nn.Linear(recurrent_output_dim, 1) if self.pooling == "attention" else None
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(recurrent_output_dim),
            nn.Linear(recurrent_output_dim, recurrent_output_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(recurrent_output_dim, model_config.embedding_dim),
        )

    def pool_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
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

        hidden, _ = self.rnn(hidden)
        if self.lstm is not None:
            hidden, _ = self.lstm(hidden)

        reconstruction = self.reconstruct_head(hidden)
        pooled = self.pool_hidden(hidden)
        embedding = self.embedding_head(pooled)
        return reconstruction, embedding


def train_recurrent_baseline(
    windows: np.ndarray,
    model_config: RecurrentBaselineConfig,
    training_config: TrainingConfig,
) -> tuple[RecurrentMaskedAutoencoder, pd.DataFrame, torch.Tensor, torch.device]:
    device = resolve_device(training_config.device)
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
        from encoder_only_transformer import sample_mask

        val_masks = sample_mask(val_windows, training_config.mask_ratio)
        val_loader = DataLoader(
            TensorDataset(val_windows, val_masks),
            batch_size=training_config.batch_size,
            shuffle=False,
        )
    else:
        val_loader = None

    model = RecurrentMaskedAutoencoder(
        input_dim=windows.shape[2],
        model_config=model_config,
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
                f"[{model_config.architecture}] epoch={epoch}/{training_config.num_epochs} "
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
            print(f"[{model_config.architecture}] early stopping at epoch {epoch}", flush=True)
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, pd.DataFrame(history), all_windows, device


def extract_recurrent_embeddings(
    model: RecurrentMaskedAutoencoder,
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


def _summarize_experiment(
    experiment_name: str,
    architecture: str,
    sequence_panel: pd.DataFrame,
    sequence_metadata: dict,
    window_size: int,
    window_end_dates: pd.Index,
    history_df: pd.DataFrame,
    cluster_scan: pd.DataFrame,
    cluster_labels: np.ndarray,
    target_cluster_count: int,
    hmm_results: dict | None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame | None, float, float]:
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

            cluster_summary = cluster_profile.groupby("cluster").agg(**agg_spec).reset_index()

    best_epoch = int(history_df.loc[history_df["val_loss"].idxmin(), "epoch"])
    best_val_loss = float(history_df["val_loss"].min())
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
    return summary, cluster_summary, comparison_table, ari_score, nmi_score


def run_recurrent_experiment(
    experiment_name: str,
    model_config: RecurrentBaselineConfig,
    data_config: SequenceDataConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    data_config = data_config or SequenceDataConfig()
    training_config = training_config or TrainingConfig()
    clustering_config = clustering_config or ClusteringConfig()
    hmm_config = hmm_config or HMMReferenceConfig()

    set_seed(training_config.random_state)

    market_data, macro_data = load_market_and_macro(data_config)
    sequence_panel, sequence_metadata = build_sequence_panel(market_data, macro_data, data_config)

    sequence_scaler = StandardScaler()
    sequence_z = pd.DataFrame(
        sequence_scaler.fit_transform(sequence_panel),
        index=sequence_panel.index,
        columns=sequence_panel.columns,
    )
    windows, window_end_dates = make_windows(sequence_z, data_config.window_size)

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

    model, history_df, all_windows, device = train_recurrent_baseline(
        windows=windows,
        model_config=model_config,
        training_config=training_config,
    )
    embeddings = extract_recurrent_embeddings(model, all_windows, device)
    cluster_scan, cluster_model, cluster_labels, target_cluster_count = cluster_embeddings(
        embeddings=embeddings,
        clustering_config=clustering_config,
        random_state=training_config.random_state,
        target_cluster_count=target_cluster_count,
    )

    summary, cluster_summary, comparison_table, _, _ = _summarize_experiment(
        experiment_name=experiment_name,
        architecture=model_config.architecture,
        sequence_panel=sequence_panel,
        sequence_metadata=sequence_metadata,
        window_size=data_config.window_size,
        window_end_dates=window_end_dates,
        history_df=history_df,
        cluster_scan=cluster_scan,
        cluster_labels=cluster_labels,
        target_cluster_count=target_cluster_count,
        hmm_results=hmm_results,
    )

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


def run_pure_rnn_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: RecurrentBaselineConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    config = model_config or RecurrentBaselineConfig(architecture="pure_rnn")
    config.architecture = "pure_rnn"
    return run_recurrent_experiment(
        experiment_name=experiment_name,
        model_config=config,
        data_config=data_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )


def run_rnn_lstm_hybrid_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: RecurrentBaselineConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    config = model_config or RecurrentBaselineConfig(architecture="rnn_lstm_hybrid")
    config.architecture = "rnn_lstm_hybrid"
    return run_recurrent_experiment(
        experiment_name=experiment_name,
        model_config=config,
        data_config=data_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )


__all__ = [
    "HMMLEARN_AVAILABLE",
    "RecurrentBaselineConfig",
    "RecurrentMaskedAutoencoder",
    "run_recurrent_experiment",
    "run_pure_rnn_experiment",
    "run_rnn_lstm_hybrid_experiment",
]
