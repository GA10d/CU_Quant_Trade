from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from encoder_only_transformer import (
    ClusteringConfig,
    EncoderOnlyTransformerConfig,
    HMMLEARN_AVAILABLE,
    HMMReferenceConfig,
    SequenceDataConfig,
    SinusoidalPositionalEncoding,
    TrainingConfig,
    cluster_embeddings,
    masked_reconstruction_loss,
    make_torch_dataloader_generator,
    resolve_device,
    run_encoder_only_experiment,
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
class ConvTransformerConfig:
    architecture: str = "conv_transformer"
    conv_channels: int = 64
    d_model: int = 64
    embedding_dim: int = 64
    n_heads: int = 4
    num_transformer_layers: int = 2
    num_conv_layers: int = 2
    kernel_size: int = 3
    dilation_base: int = 2
    dropout: float = 0.10
    feedforward_multiplier: float = 2.0
    pooling: str = "attention"
    use_mask_embedding: bool = True


@dataclass
class Seq2SeqTransformerConfig:
    architecture: str = "mae_transformer"
    d_model: int = 64
    embedding_dim: int = 64
    n_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 1
    dropout: float = 0.10
    feedforward_multiplier: float = 2.0
    pooling: str = "mean"
    use_mask_embedding: bool = True


class TemporalBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(output_dim, output_dim, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(input_dim, output_dim, kernel_size=1) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual(x)


class ConvTransformerAutoencoder(nn.Module):
    def __init__(self, input_dim: int, model_config: ConvTransformerConfig, max_len: int):
        super().__init__()
        if model_config.architecture not in {"conv_transformer", "tcn_transformer"}:
            raise ValueError("architecture must be one of: conv_transformer, tcn_transformer")
        if model_config.pooling not in {"attention", "mean", "last"}:
            raise ValueError("pooling must be one of: attention, mean, last")
        if model_config.d_model % model_config.n_heads != 0:
            raise ValueError(
                f"d_model={model_config.d_model} must be divisible by n_heads={model_config.n_heads}."
            )

        self.architecture = model_config.architecture
        self.pooling = model_config.pooling
        self.use_mask_embedding = model_config.use_mask_embedding

        self.input_norm = nn.LayerNorm(input_dim)
        self.mask_proj = nn.Linear(input_dim, model_config.d_model, bias=False)

        if self.architecture == "conv_transformer":
            conv_layers = []
            in_channels = input_dim
            for _ in range(model_config.num_conv_layers):
                conv_layers.extend(
                    [
                        nn.Conv1d(
                            in_channels,
                            model_config.conv_channels,
                            kernel_size=model_config.kernel_size,
                            padding=(model_config.kernel_size - 1) // 2,
                        ),
                        nn.GELU(),
                        nn.Dropout(model_config.dropout),
                    ]
                )
                in_channels = model_config.conv_channels
            self.conv_stem = nn.Sequential(*conv_layers)
        else:
            blocks = []
            in_channels = input_dim
            for layer_idx in range(model_config.num_conv_layers):
                dilation = model_config.dilation_base**layer_idx
                blocks.append(
                    TemporalBlock(
                        input_dim=in_channels,
                        output_dim=model_config.conv_channels,
                        kernel_size=model_config.kernel_size,
                        dilation=dilation,
                        dropout=model_config.dropout,
                    )
                )
                in_channels = model_config.conv_channels
            self.conv_stem = nn.Sequential(*blocks)

        self.input_proj = nn.Linear(model_config.conv_channels, model_config.d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(model_config.d_model, max_len=max_len)
        self.pos_dropout = nn.Dropout(model_config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config.d_model,
            nhead=model_config.n_heads,
            dim_feedforward=int(model_config.d_model * model_config.feedforward_multiplier),
            dropout=model_config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config.num_transformer_layers,
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
        if self.pooling == "attention":
            assert self.pool_score is not None
            weights = torch.softmax(self.pool_score(hidden), dim=1)
            return (hidden * weights).sum(dim=1)
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        return hidden[:, -1, :]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_x = self.input_norm(x)
        conv_input = normalized_x.transpose(1, 2)
        conv_hidden = self.conv_stem(conv_input).transpose(1, 2)
        hidden = self.input_proj(conv_hidden)
        if mask is not None and self.use_mask_embedding:
            hidden = hidden + self.mask_proj(mask.float())
        hidden = self.pos_dropout(self.pos_encoder(hidden))
        hidden = self.encoder(hidden)
        reconstruction = self.reconstruct_head(hidden)
        embedding = self.embedding_head(self.pool_hidden(hidden))
        return reconstruction, embedding


class EncoderDecoderTransformerAutoencoder(nn.Module):
    def __init__(self, input_dim: int, model_config: Seq2SeqTransformerConfig, max_len: int):
        super().__init__()
        if model_config.architecture not in {"mae_transformer", "transformer_autoencoder"}:
            raise ValueError("architecture must be one of: mae_transformer, transformer_autoencoder")
        if model_config.pooling not in {"attention", "mean", "last"}:
            raise ValueError("pooling must be one of: attention, mean, last")
        if model_config.d_model % model_config.n_heads != 0:
            raise ValueError(
                f"d_model={model_config.d_model} must be divisible by n_heads={model_config.n_heads}."
            )

        self.architecture = model_config.architecture
        self.pooling = model_config.pooling
        self.use_mask_embedding = model_config.use_mask_embedding

        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder_input_proj = nn.Linear(input_dim, model_config.d_model)
        self.decoder_input_proj = nn.Linear(input_dim, model_config.d_model)
        self.mask_proj = nn.Linear(input_dim, model_config.d_model, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model_config.d_model))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        self.encoder_pos = SinusoidalPositionalEncoding(model_config.d_model, max_len=max_len)
        self.decoder_pos = SinusoidalPositionalEncoding(model_config.d_model, max_len=max_len)
        self.pos_dropout = nn.Dropout(model_config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config.d_model,
            nhead=model_config.n_heads,
            dim_feedforward=int(model_config.d_model * model_config.feedforward_multiplier),
            dropout=model_config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config.num_encoder_layers,
            norm=nn.LayerNorm(model_config.d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_config.d_model,
            nhead=model_config.n_heads,
            dim_feedforward=int(model_config.d_model * model_config.feedforward_multiplier),
            dropout=model_config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=model_config.num_decoder_layers,
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

    def pool_hidden(self, hidden: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        if token_mask is not None:
            valid_mask = (~token_mask).float().unsqueeze(-1)
        else:
            valid_mask = None

        if self.pooling == "attention":
            assert self.pool_score is not None
            scores = self.pool_score(hidden)
            if valid_mask is not None:
                scores = scores.masked_fill(valid_mask == 0, -1e9)
            weights = torch.softmax(scores, dim=1)
            return (hidden * weights).sum(dim=1)

        if self.pooling == "mean":
            if valid_mask is None:
                return hidden.mean(dim=1)
            denom = valid_mask.sum(dim=1).clamp_min(1.0)
            return (hidden * valid_mask).sum(dim=1) / denom

        if token_mask is None:
            return hidden[:, -1, :]

        lengths = (~token_mask).sum(dim=1).clamp_min(1)
        indices = (lengths - 1).long()
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_indices, indices]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_x = self.input_norm(x)
        encoder_tokens = self.encoder_input_proj(normalized_x)
        token_mask = None

        if mask is not None:
            token_mask = mask.any(dim=2)
            if token_mask.all(dim=1).any():
                token_mask = token_mask.clone()
                fully_masked = token_mask.all(dim=1)
                token_mask[fully_masked, 0] = False
            if self.use_mask_embedding:
                encoder_tokens = encoder_tokens + self.mask_proj(mask.float())

        encoder_hidden = self.pos_dropout(self.encoder_pos(encoder_tokens))
        memory = self.encoder(encoder_hidden, src_key_padding_mask=token_mask)

        if self.architecture == "mae_transformer":
            decoder_tokens = self.decoder_input_proj(normalized_x)
            if token_mask is not None:
                mask_tokens = self.mask_token.expand(x.size(0), x.size(1), -1)
                decoder_tokens = torch.where(token_mask.unsqueeze(-1), mask_tokens, decoder_tokens)
        else:
            decoder_tokens = self.decoder_input_proj(normalized_x)

        decoder_hidden = self.decoder(
            tgt=self.pos_dropout(self.decoder_pos(decoder_tokens)),
            memory=memory,
            memory_key_padding_mask=token_mask,
        )
        reconstruction = self.reconstruct_head(decoder_hidden)
        embedding = self.embedding_head(self.pool_hidden(memory, token_mask=token_mask))
        return reconstruction, embedding


def _train_masked_model(
    windows: np.ndarray,
    model: nn.Module,
    training_config: TrainingConfig,
    log_prefix: str,
    split_indices: dict[str, np.ndarray] | None = None,
) -> tuple[nn.Module, pd.DataFrame, torch.Tensor, torch.device]:
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
    from encoder_only_transformer import sample_mask

    if len(val_windows) > 0:
        val_masks = sample_mask(val_windows, training_config.mask_ratio)
        val_loader = DataLoader(
            TensorDataset(val_windows, val_masks),
            batch_size=training_config.batch_size,
            shuffle=False,
        )
    else:
        val_loader = None

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


def _train_full_reconstruction_model(
    windows: np.ndarray,
    model: nn.Module,
    training_config: TrainingConfig,
    log_prefix: str,
    split_indices: dict[str, np.ndarray] | None = None,
) -> tuple[nn.Module, pd.DataFrame, torch.Tensor, torch.device]:
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


def _extract_embeddings(model: nn.Module, all_windows: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(all_windows), batch_size=128, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            _, embedding = model(batch)
            embeddings.append(embedding.cpu().numpy())
    return np.vstack(embeddings)


def run_cls_token_transformer_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: EncoderOnlyTransformerConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    config = replace(model_config or EncoderOnlyTransformerConfig(), pooling="cls")
    return run_encoder_only_experiment(
        experiment_name=experiment_name,
        data_config=data_config,
        model_config=config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )


def _run_transformer_variant_experiment(
    experiment_name: str,
    architecture: str,
    model_config: ConvTransformerConfig | Seq2SeqTransformerConfig,
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
    prepared_inputs = prepare_sequence_experiment_inputs(data_config, training_config=training_config)
    hmm_results = prepare_hmm_reference_for_experiment(
        data_config=data_config,
        training_config=training_config,
        hmm_config=hmm_config,
        market_data=prepared_inputs["market_data"],
        macro_data=prepared_inputs["macro_data"],
    )
    target_cluster_count = resolve_target_cluster_count(clustering_config, hmm_results)

    if architecture in {"conv_transformer", "tcn_transformer"}:
        model = ConvTransformerAutoencoder(
            input_dim=prepared_inputs["windows"].shape[2],
            model_config=model_config,
            max_len=data_config.window_size,
        )
        model, history_df, all_windows, device = _train_masked_model(
            windows=prepared_inputs["windows"],
            model=model,
            training_config=training_config,
            log_prefix=architecture,
            split_indices=prepared_inputs["window_split_indices"],
        )
    elif architecture == "mae_transformer":
        model = EncoderDecoderTransformerAutoencoder(
            input_dim=prepared_inputs["windows"].shape[2],
            model_config=model_config,
            max_len=data_config.window_size,
        )
        model, history_df, all_windows, device = _train_masked_model(
            windows=prepared_inputs["windows"],
            model=model,
            training_config=training_config,
            log_prefix=architecture,
            split_indices=prepared_inputs["window_split_indices"],
        )
    elif architecture == "transformer_autoencoder":
        model = EncoderDecoderTransformerAutoencoder(
            input_dim=prepared_inputs["windows"].shape[2],
            model_config=model_config,
            max_len=data_config.window_size,
        )
        model, history_df, all_windows, device = _train_full_reconstruction_model(
            windows=prepared_inputs["windows"],
            model=model,
            training_config=training_config,
            log_prefix=architecture,
            split_indices=prepared_inputs["window_split_indices"],
        )
    else:
        raise ValueError(f"Unsupported transformer architecture: {architecture}")

    embeddings = _extract_embeddings(model, all_windows, device)
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


def run_conv_transformer_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: ConvTransformerConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    config = model_config or ConvTransformerConfig(architecture="conv_transformer")
    config.architecture = "conv_transformer"
    return _run_transformer_variant_experiment(
        experiment_name=experiment_name,
        architecture="conv_transformer",
        model_config=config,
        data_config=data_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )


def run_tcn_transformer_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: ConvTransformerConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    config = model_config or ConvTransformerConfig(architecture="tcn_transformer")
    config.architecture = "tcn_transformer"
    return _run_transformer_variant_experiment(
        experiment_name=experiment_name,
        architecture="tcn_transformer",
        model_config=config,
        data_config=data_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )


def run_mae_transformer_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: Seq2SeqTransformerConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    config = model_config or Seq2SeqTransformerConfig(architecture="mae_transformer")
    config.architecture = "mae_transformer"
    return _run_transformer_variant_experiment(
        experiment_name=experiment_name,
        architecture="mae_transformer",
        model_config=config,
        data_config=data_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )


def run_transformer_autoencoder_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: Seq2SeqTransformerConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    config = model_config or Seq2SeqTransformerConfig(architecture="transformer_autoencoder")
    config.architecture = "transformer_autoencoder"
    return _run_transformer_variant_experiment(
        experiment_name=experiment_name,
        architecture="transformer_autoencoder",
        model_config=config,
        data_config=data_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )


__all__ = [
    "HMMLEARN_AVAILABLE",
    "ConvTransformerConfig",
    "Seq2SeqTransformerConfig",
    "run_cls_token_transformer_experiment",
    "run_conv_transformer_experiment",
    "run_tcn_transformer_experiment",
    "run_mae_transformer_experiment",
    "run_transformer_autoencoder_experiment",
]
