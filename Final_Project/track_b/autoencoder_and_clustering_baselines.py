from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from encoder_only_transformer import (
    ClusteringConfig,
    HMMLEARN_AVAILABLE,
    HMMReferenceConfig,
    SequenceDataConfig,
    TrainingConfig,
    cluster_embeddings,
    make_torch_dataloader_generator,
    resolve_device,
    set_seed,
    split_ordered_train_random_holdout_indices,
)
from experiment_utils import (
    build_experiment_result,
    prepare_hmm_reference_for_experiment,
    prepare_sequence_experiment_inputs,
    resolve_target_cluster_count,
    summarize_experiment_outputs,
)


@dataclass
class SequenceAutoencoderConfig:
    architecture: str = "rnn_autoencoder"
    input_projection_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 2
    embedding_dim: int = 32
    decoder_hidden_dim: int = 64
    dropout: float = 0.10
    nonlinearity: str = "tanh"
    bidirectional: bool = False


@dataclass
class WindowVAEConfig:
    architecture: str = "vae"
    hidden_dim: int = 256
    latent_dim: int = 32
    beta: float = 1e-3
    dropout: float = 0.10


@dataclass
class ClassicalClusteringConfig:
    architecture: str = "kmeans_baseline"
    pca_components: int = 16


class SequenceAutoencoder(nn.Module):
    def __init__(self, input_dim: int, model_config: SequenceAutoencoderConfig):
        super().__init__()
        if model_config.architecture not in {"rnn_autoencoder", "lstm_autoencoder"}:
            raise ValueError("architecture must be one of: rnn_autoencoder, lstm_autoencoder")

        self.architecture = model_config.architecture
        self.bidirectional = model_config.bidirectional

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, model_config.input_projection_dim)

        recurrent_dropout = model_config.dropout if model_config.num_layers > 1 else 0.0
        if self.architecture == "rnn_autoencoder":
            self.encoder = nn.RNN(
                input_size=model_config.input_projection_dim,
                hidden_size=model_config.hidden_dim,
                num_layers=model_config.num_layers,
                nonlinearity=model_config.nonlinearity,
                dropout=recurrent_dropout,
                batch_first=True,
                bidirectional=model_config.bidirectional,
            )
            self.decoder = nn.RNN(
                input_size=model_config.embedding_dim,
                hidden_size=model_config.decoder_hidden_dim,
                num_layers=model_config.num_layers,
                nonlinearity=model_config.nonlinearity,
                dropout=recurrent_dropout,
                batch_first=True,
                bidirectional=model_config.bidirectional,
            )
        else:
            self.encoder = nn.LSTM(
                input_size=model_config.input_projection_dim,
                hidden_size=model_config.hidden_dim,
                num_layers=model_config.num_layers,
                dropout=recurrent_dropout,
                batch_first=True,
                bidirectional=model_config.bidirectional,
            )
            self.decoder = nn.LSTM(
                input_size=model_config.embedding_dim,
                hidden_size=model_config.decoder_hidden_dim,
                num_layers=model_config.num_layers,
                dropout=recurrent_dropout,
                batch_first=True,
                bidirectional=model_config.bidirectional,
            )

        encoder_output_dim = model_config.hidden_dim * (2 if model_config.bidirectional else 1)
        decoder_output_dim = model_config.decoder_hidden_dim * (2 if model_config.bidirectional else 1)

        self.embedding_head = nn.Sequential(
            nn.LayerNorm(encoder_output_dim),
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(encoder_output_dim, model_config.embedding_dim),
        )
        self.reconstruct_head = nn.Sequential(
            nn.LayerNorm(decoder_output_dim),
            nn.Linear(decoder_output_dim, decoder_output_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(decoder_output_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_x = self.input_norm(x)
        encoded_input = self.input_proj(normalized_x)
        _, hidden_state = self.encoder(encoded_input)

        if self.architecture == "lstm_autoencoder":
            hidden_tensor = hidden_state[0]
        else:
            hidden_tensor = hidden_state

        if self.bidirectional:
            context = torch.cat([hidden_tensor[-2], hidden_tensor[-1]], dim=-1)
        else:
            context = hidden_tensor[-1]

        embedding = self.embedding_head(context)
        decoder_seed = embedding.unsqueeze(1).expand(-1, x.size(1), -1)
        decoded_sequence, _ = self.decoder(decoder_seed)
        reconstruction = self.reconstruct_head(decoded_sequence)
        return reconstruction, embedding


class WindowVAE(nn.Module):
    def __init__(self, window_size: int, input_dim: int, model_config: WindowVAEConfig):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        flat_dim = window_size * input_dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, model_config.hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(model_config.hidden_dim, model_config.latent_dim)
        self.logvar_head = nn.Linear(model_config.hidden_dim, model_config.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(model_config.latent_dim, model_config.hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
            nn.GELU(),
            nn.Linear(model_config.hidden_dim, flat_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_x = x.reshape(x.size(0), -1)
        hidden = self.encoder(flat_x)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        flat_reconstruction = self.decoder(z)
        reconstruction = flat_reconstruction.reshape(x.size(0), self.window_size, self.input_dim)
        return reconstruction, mu, mu, logvar


def _train_full_autoencoder(
    windows: np.ndarray,
    model: nn.Module,
    training_config: TrainingConfig,
    log_prefix: str,
) -> tuple[nn.Module, pd.DataFrame, torch.Tensor, torch.device]:
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
    val_loader = (
        DataLoader(TensorDataset(val_windows), batch_size=training_config.batch_size, shuffle=False)
        if len(val_windows) > 0
        else None
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history: list[dict] = []
    best_state_dict = None
    best_val_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(1, training_config.num_epochs + 1):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction, _ = model(batch)
            loss = torch.mean((reconstruction - batch) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    reconstruction, _ = model(batch)
                    val_losses.append(float(torch.mean((reconstruction - batch) ** 2).item()))
            val_loss_value = float(np.mean(val_losses))
        else:
            val_loss_value = train_loss

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss_value})
        if training_config.log_every and (epoch % training_config.log_every == 0 or epoch == training_config.num_epochs):
            print(
                f"[{log_prefix}] epoch={epoch}/{training_config.num_epochs} "
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
            print(f"[{log_prefix}] early stopping at epoch {epoch}", flush=True)
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, pd.DataFrame(history), all_windows, device


def _train_vae(
    windows: np.ndarray,
    model: WindowVAE,
    model_config: WindowVAEConfig,
    training_config: TrainingConfig,
) -> tuple[WindowVAE, pd.DataFrame, torch.Tensor, torch.device]:
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
    val_loader = (
        DataLoader(TensorDataset(val_windows), batch_size=training_config.batch_size, shuffle=False)
        if len(val_windows) > 0
        else None
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history: list[dict] = []
    best_state_dict = None
    best_val_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(1, training_config.num_epochs + 1):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction, embedding, mu, logvar = model(batch)
            recon_loss = torch.mean((reconstruction - batch) ** 2)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + model_config.beta * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    reconstruction, _, mu, logvar = model(batch)
                    recon_loss = torch.mean((reconstruction - batch) ** 2)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    val_losses.append(float((recon_loss + model_config.beta * kl_loss).item()))
            val_loss_value = float(np.mean(val_losses))
        else:
            val_loss_value = train_loss

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss_value})
        if training_config.log_every and (epoch % training_config.log_every == 0 or epoch == training_config.num_epochs):
            print(
                f"[vae] epoch={epoch}/{training_config.num_epochs} "
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
            print(f"[vae] early stopping at epoch {epoch}", flush=True)
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, pd.DataFrame(history), all_windows, device


def _extract_autoencoder_embeddings(model: nn.Module, all_windows: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(all_windows), batch_size=128, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            outputs = model(batch)
            if len(outputs) == 2:
                _, embedding = outputs
            else:
                _, embedding, _, _ = outputs
            embeddings.append(embedding.cpu().numpy())
    return np.vstack(embeddings)


def _run_autoencoder_experiment(
    experiment_name: str,
    architecture: str,
    model_config: SequenceAutoencoderConfig | WindowVAEConfig,
    data_config: SequenceDataConfig | None,
    training_config: TrainingConfig | None,
    clustering_config: ClusteringConfig | None,
    hmm_config: HMMReferenceConfig | None,
) -> dict:
    data_config = data_config or SequenceDataConfig()
    training_config = training_config or TrainingConfig()
    clustering_config = clustering_config or ClusteringConfig()
    hmm_config = hmm_config or HMMReferenceConfig()

    set_seed(training_config.random_state)
    prepared_inputs = prepare_sequence_experiment_inputs(data_config)
    hmm_results = prepare_hmm_reference_for_experiment(
        data_config=data_config,
        training_config=training_config,
        hmm_config=hmm_config,
        market_data=prepared_inputs["market_data"],
        macro_data=prepared_inputs["macro_data"],
    )
    target_cluster_count = resolve_target_cluster_count(clustering_config, hmm_results)

    if architecture in {"rnn_autoencoder", "lstm_autoencoder"}:
        model = SequenceAutoencoder(input_dim=prepared_inputs["windows"].shape[2], model_config=model_config)
        model, history_df, all_windows, device = _train_full_autoencoder(
            windows=prepared_inputs["windows"],
            model=model,
            training_config=training_config,
            log_prefix=architecture,
        )
    elif architecture == "vae":
        model = WindowVAE(
            window_size=data_config.window_size,
            input_dim=prepared_inputs["windows"].shape[2],
            model_config=model_config,
        )
        model, history_df, all_windows, device = _train_vae(
            windows=prepared_inputs["windows"],
            model=model,
            model_config=model_config,
            training_config=training_config,
        )
    else:
        raise ValueError(f"Unsupported autoencoder architecture: {architecture}")

    embeddings = _extract_autoencoder_embeddings(model, all_windows, device)
    cluster_scan, cluster_model, cluster_labels, target_cluster_count = cluster_embeddings(
        embeddings=embeddings,
        clustering_config=clustering_config,
        random_state=training_config.random_state,
        target_cluster_count=target_cluster_count,
    )

    summary, cluster_summary, comparison_table = summarize_experiment_outputs(
        experiment_name=experiment_name,
        architecture=architecture,
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
    return build_experiment_result(
        experiment_name=experiment_name,
        summary=summary,
        data_config=data_config,
        model_config=model_config,
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


def run_rnn_autoencoder_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: SequenceAutoencoderConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    config = model_config or SequenceAutoencoderConfig(architecture="rnn_autoencoder")
    config.architecture = "rnn_autoencoder"
    return _run_autoencoder_experiment(
        experiment_name=experiment_name,
        architecture="rnn_autoencoder",
        model_config=config,
        data_config=data_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )


def run_lstm_autoencoder_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: SequenceAutoencoderConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    config = model_config or SequenceAutoencoderConfig(architecture="lstm_autoencoder")
    config.architecture = "lstm_autoencoder"
    return _run_autoencoder_experiment(
        experiment_name=experiment_name,
        architecture="lstm_autoencoder",
        model_config=config,
        data_config=data_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )


def run_vae_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: WindowVAEConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    config = model_config or WindowVAEConfig()
    return _run_autoencoder_experiment(
        experiment_name=experiment_name,
        architecture="vae",
        model_config=config,
        data_config=data_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )


def _run_classical_clustering_experiment(
    experiment_name: str,
    architecture: str,
    model_config: ClassicalClusteringConfig,
    data_config: SequenceDataConfig | None,
    training_config: TrainingConfig | None,
    clustering_config: ClusteringConfig | None,
    hmm_config: HMMReferenceConfig | None,
) -> dict:
    data_config = data_config or SequenceDataConfig()
    training_config = training_config or TrainingConfig()
    clustering_config = clustering_config or ClusteringConfig()
    hmm_config = hmm_config or HMMReferenceConfig()

    set_seed(training_config.random_state)
    prepared_inputs = prepare_sequence_experiment_inputs(data_config)
    hmm_results = prepare_hmm_reference_for_experiment(
        data_config=data_config,
        training_config=training_config,
        hmm_config=hmm_config,
        market_data=prepared_inputs["market_data"],
        macro_data=prepared_inputs["macro_data"],
    )
    target_cluster_count = resolve_target_cluster_count(clustering_config, hmm_results)

    raw_embeddings = prepared_inputs["windows"].reshape(len(prepared_inputs["windows"]), -1)
    fitted_model: dict[str, object] = {}

    if architecture == "kmeans_baseline":
        embeddings = raw_embeddings
    elif architecture == "pca_kmeans_baseline":
        n_components = min(
            model_config.pca_components,
            raw_embeddings.shape[0] - 1,
            raw_embeddings.shape[1],
        )
        n_components = max(2, n_components)
        pca_model = PCA(n_components=n_components, random_state=training_config.random_state)
        embeddings = pca_model.fit_transform(raw_embeddings)
        fitted_model["pca_model"] = pca_model
    else:
        raise ValueError(f"Unsupported classical clustering architecture: {architecture}")

    cluster_scan, cluster_model, cluster_labels, target_cluster_count = cluster_embeddings(
        embeddings=embeddings,
        clustering_config=clustering_config,
        random_state=training_config.random_state,
        target_cluster_count=target_cluster_count,
    )

    history_df = pd.DataFrame([{"epoch": 0, "train_loss": np.nan, "val_loss": np.nan}])
    summary, cluster_summary, comparison_table = summarize_experiment_outputs(
        experiment_name=experiment_name,
        architecture=architecture,
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
    return build_experiment_result(
        experiment_name=experiment_name,
        summary=summary,
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
        prepared_inputs=prepared_inputs,
        model=fitted_model if fitted_model else None,
        history_df=history_df,
        embeddings=embeddings,
        cluster_scan=cluster_scan,
        cluster_model=cluster_model,
        cluster_labels=cluster_labels,
        cluster_summary=cluster_summary,
        comparison_table=comparison_table,
        hmm_results=hmm_results,
        device="cpu",
    )


def run_kmeans_baseline_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: ClassicalClusteringConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    config = model_config or ClassicalClusteringConfig(architecture="kmeans_baseline")
    config.architecture = "kmeans_baseline"
    return _run_classical_clustering_experiment(
        experiment_name=experiment_name,
        architecture="kmeans_baseline",
        model_config=config,
        data_config=data_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )


def run_pca_kmeans_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: ClassicalClusteringConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    config = model_config or ClassicalClusteringConfig(architecture="pca_kmeans_baseline")
    config.architecture = "pca_kmeans_baseline"
    return _run_classical_clustering_experiment(
        experiment_name=experiment_name,
        architecture="pca_kmeans_baseline",
        model_config=config,
        data_config=data_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )


__all__ = [
    "HMMLEARN_AVAILABLE",
    "SequenceAutoencoderConfig",
    "WindowVAEConfig",
    "ClassicalClusteringConfig",
    "run_rnn_autoencoder_experiment",
    "run_lstm_autoencoder_experiment",
    "run_vae_experiment",
    "run_kmeans_baseline_experiment",
    "run_pca_kmeans_experiment",
]
