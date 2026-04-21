from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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
        if model_config.architecture not in {"mae_transformer", "transformer_autoencoder", "softmax_regime_transformer"}:
            raise ValueError(
                "architecture must be one of: mae_transformer, transformer_autoencoder, softmax_regime_transformer"
            )
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


class SoftmaxBottleneckTransformerAutoencoder(nn.Module):
    def __init__(self, input_dim: int, model_config: Seq2SeqTransformerConfig, max_len: int):
        super().__init__()
        if model_config.architecture != "softmax_regime_transformer":
            raise ValueError("architecture must be softmax_regime_transformer")
        if model_config.pooling not in {"attention", "mean", "last"}:
            raise ValueError("pooling must be one of: attention, mean, last")
        if model_config.d_model % model_config.n_heads != 0:
            raise ValueError(
                f"d_model={model_config.d_model} must be divisible by n_heads={model_config.n_heads}."
            )

        self.pooling = model_config.pooling
        self.use_mask_embedding = model_config.use_mask_embedding
        self.latent_dim = model_config.embedding_dim
        self.max_len = max_len

        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder_input_proj = nn.Linear(input_dim, model_config.d_model)
        self.mask_proj = nn.Linear(input_dim, model_config.d_model, bias=False)
        self.encoder_pos = SinusoidalPositionalEncoding(model_config.d_model, max_len=max_len)
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

        self.pool_score = nn.Linear(model_config.d_model, 1) if self.pooling == "attention" else None
        self.latent_head = nn.Sequential(
            nn.LayerNorm(model_config.d_model),
            nn.Linear(model_config.d_model, model_config.d_model),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.d_model, model_config.embedding_dim),
        )

        self.decoder_latent_proj = nn.Linear(model_config.embedding_dim, model_config.d_model)
        self.decoder_queries = nn.Parameter(torch.zeros(1, max_len, model_config.d_model))
        nn.init.normal_(self.decoder_queries, mean=0.0, std=0.02)
        self.decoder_pos = SinusoidalPositionalEncoding(model_config.d_model, max_len=max_len)

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

    def pool_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.pooling == "attention":
            assert self.pool_score is not None
            weights = torch.softmax(self.pool_score(hidden), dim=1)
            return (hidden * weights).sum(dim=1)
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        return hidden[:, -1, :]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_x = self.input_norm(x)
        encoder_tokens = self.encoder_input_proj(normalized_x)
        if mask is not None and self.use_mask_embedding:
            encoder_tokens = encoder_tokens + self.mask_proj(mask.float())

        encoder_hidden = self.pos_dropout(self.encoder_pos(encoder_tokens))
        memory = self.encoder(encoder_hidden)
        latent_logits = self.latent_head(self.pool_hidden(memory))
        latent_probs = torch.softmax(latent_logits, dim=-1)

        latent_memory = self.decoder_latent_proj(latent_probs).unsqueeze(1)
        decoder_queries = self.decoder_queries[:, : x.size(1), :].expand(x.size(0), -1, -1)
        decoder_queries = self.pos_dropout(self.decoder_pos(decoder_queries))
        decoder_hidden = self.decoder(tgt=decoder_queries, memory=latent_memory)
        reconstruction = self.reconstruct_head(decoder_hidden)
        return reconstruction, latent_logits, latent_probs


class PcaSoftmaxBottleneckTransformerAutoencoder(nn.Module):
    def __init__(self, input_dim: int, model_config: Seq2SeqTransformerConfig, max_len: int):
        super().__init__()
        if model_config.architecture != "pca_softmax_regime_transformer":
            raise ValueError("architecture must be pca_softmax_regime_transformer")
        if model_config.pooling not in {"attention", "mean", "last"}:
            raise ValueError("pooling must be one of: attention, mean, last")
        if model_config.d_model % model_config.n_heads != 0:
            raise ValueError(
                f"d_model={model_config.d_model} must be divisible by n_heads={model_config.n_heads}."
            )

        self.pooling = model_config.pooling
        self.use_mask_embedding = model_config.use_mask_embedding
        self.latent_dim = model_config.embedding_dim
        self.max_len = max_len

        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder_input_proj = nn.Linear(input_dim, model_config.d_model)
        self.mask_proj = nn.Linear(input_dim, model_config.d_model, bias=False)
        self.encoder_pos = SinusoidalPositionalEncoding(model_config.d_model, max_len=max_len)
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

        self.pool_score = nn.Linear(model_config.d_model, 1) if self.pooling == "attention" else None
        self.latent_head = nn.Sequential(
            nn.LayerNorm(model_config.d_model),
            nn.Linear(model_config.d_model, model_config.d_model),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.d_model, model_config.embedding_dim),
        )

        self.decoder_latent_proj = nn.Linear(model_config.embedding_dim, model_config.d_model)
        self.decoder_queries = nn.Parameter(torch.zeros(1, max_len, model_config.d_model))
        nn.init.normal_(self.decoder_queries, mean=0.0, std=0.02)
        self.decoder_pos = SinusoidalPositionalEncoding(model_config.d_model, max_len=max_len)

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
        encoder_tokens = self.encoder_input_proj(normalized_x)
        if mask is not None and self.use_mask_embedding:
            encoder_tokens = encoder_tokens + self.mask_proj(mask.float())

        encoder_hidden = self.pos_dropout(self.encoder_pos(encoder_tokens))
        memory = self.encoder(encoder_hidden)
        latent_embedding = self.latent_head(self.pool_hidden(memory))

        latent_memory = self.decoder_latent_proj(latent_embedding).unsqueeze(1)
        decoder_queries = self.decoder_queries[:, : x.size(1), :].expand(x.size(0), -1, -1)
        decoder_queries = self.pos_dropout(self.decoder_pos(decoder_queries))
        decoder_hidden = self.decoder(tgt=decoder_queries, memory=latent_memory)
        reconstruction = self.reconstruct_head(decoder_hidden)
        return reconstruction, latent_embedding


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


def _extract_softmax_regime_probabilities(
    model: SoftmaxBottleneckTransformerAutoencoder,
    all_windows: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(all_windows), batch_size=128, shuffle=False)
    probabilities = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            _, _, batch_probabilities = model(batch)
            probabilities.append(batch_probabilities.cpu().numpy())
    return np.vstack(probabilities)


def _build_softmax_regime_outputs(
    probabilities: np.ndarray,
    target_cluster_count: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object], np.ndarray]:
    if probabilities.ndim != 2:
        raise ValueError(f"Expected 2D probabilities for softmax regime output, got shape={probabilities.shape}.")
    if probabilities.shape[1] != target_cluster_count:
        raise ValueError(
            "softmax_regime_transformer requires the encoder embedding dimension to match "
            f"target_cluster_count={target_cluster_count}, got embedding_dim={probabilities.shape[1]}."
        )

    cluster_labels = probabilities.argmax(axis=1).astype(int)

    silhouette_value = np.nan
    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) > 1 and len(unique_labels) < len(probabilities):
        try:
            silhouette_value = float(silhouette_score(probabilities, cluster_labels))
        except Exception:
            silhouette_value = np.nan

    cluster_scan = pd.DataFrame(
        [{"n_clusters": int(target_cluster_count), "silhouette": silhouette_value, "selection_method": "softmax_argmax"}]
    )
    cluster_model = {
        "selection_method": "softmax_argmax",
        "target_cluster_count": int(target_cluster_count),
        "active_cluster_count": int(len(unique_labels)),
    }
    return probabilities, cluster_scan, cluster_model, cluster_labels


def _build_pca_softmax_regime_outputs(
    embeddings: np.ndarray,
    train_indices: np.ndarray,
    target_cluster_count: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object], np.ndarray]:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings for PCA-softmax regime output, got shape={embeddings.shape}.")
    if embeddings.shape[1] < target_cluster_count:
        raise ValueError(
            "pca_softmax_regime_transformer requires latent_dim >= target_cluster_count, "
            f"got latent_dim={embeddings.shape[1]} and target_cluster_count={target_cluster_count}."
        )

    pca_model = PCA(n_components=target_cluster_count)
    pca_model.fit(embeddings[train_indices])
    pca_embeddings = pca_model.transform(embeddings)

    shifted = pca_embeddings - pca_embeddings.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
    cluster_labels = probabilities.argmax(axis=1).astype(int)

    silhouette_value = np.nan
    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) > 1 and len(unique_labels) < len(probabilities):
        try:
            silhouette_value = float(silhouette_score(probabilities, cluster_labels))
        except Exception:
            silhouette_value = np.nan

    cluster_scan = pd.DataFrame(
        [{"n_clusters": int(target_cluster_count), "silhouette": silhouette_value, "selection_method": "pca_softmax_argmax"}]
    )
    cluster_model = {
        "selection_method": "pca_softmax_argmax",
        "target_cluster_count": int(target_cluster_count),
        "active_cluster_count": int(len(unique_labels)),
        "explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
    }
    return probabilities, cluster_scan, cluster_model, cluster_labels


def _train_softmax_bottleneck_model(
    windows: np.ndarray,
    model: SoftmaxBottleneckTransformerAutoencoder,
    training_config: TrainingConfig,
    log_prefix: str,
    split_indices: dict[str, np.ndarray] | None = None,
    balance_weight: float = 0.10,
    confidence_weight: float = 0.01,
) -> tuple[SoftmaxBottleneckTransformerAutoencoder, pd.DataFrame, torch.Tensor, torch.device]:
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

    uniform_target = None
    history: list[dict] = []
    best_state_dict = None
    best_val_loss = float("inf")
    no_improve_epochs = 0

    def compute_loss(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction, _, probabilities = model(batch)
        recon_loss = torch.mean((reconstruction - batch) ** 2)
        probs = probabilities.clamp_min(1e-8)

        nonlocal uniform_target
        if uniform_target is None or uniform_target.numel() != probs.size(1):
            uniform_target = torch.full((probs.size(1),), 1.0 / probs.size(1), device=probs.device)

        mean_probs = probs.mean(dim=0)
        balance_loss = torch.sum(mean_probs * (torch.log(mean_probs) - torch.log(uniform_target)))
        confidence_loss = -(probs * torch.log(probs)).sum(dim=1).mean()
        loss = recon_loss + balance_weight * balance_loss + confidence_weight * confidence_loss
        return loss, recon_loss, balance_loss, confidence_loss

    for epoch in range(1, training_config.num_epochs + 1):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, _, _, _ = compute_loss(batch)
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
                    val_loss, _, _, _ = compute_loss(batch)
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
    elif architecture == "softmax_regime_transformer":
        model = SoftmaxBottleneckTransformerAutoencoder(
            input_dim=prepared_inputs["windows"].shape[2],
            model_config=model_config,
            max_len=data_config.window_size,
        )
        model, history_df, all_windows, device = _train_softmax_bottleneck_model(
            windows=prepared_inputs["windows"],
            model=model,
            training_config=training_config,
            log_prefix=architecture,
            split_indices=prepared_inputs["window_split_indices"],
        )
    elif architecture == "pca_softmax_regime_transformer":
        model = PcaSoftmaxBottleneckTransformerAutoencoder(
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

    if architecture == "softmax_regime_transformer":
        if target_cluster_count is None:
            target_cluster_count = 4
        raw_probabilities = _extract_softmax_regime_probabilities(model, all_windows, device)
        embeddings, cluster_scan, cluster_model, cluster_labels = _build_softmax_regime_outputs(
            probabilities=raw_probabilities,
            target_cluster_count=int(target_cluster_count),
        )
    elif architecture == "pca_softmax_regime_transformer":
        if target_cluster_count is None:
            target_cluster_count = 4
        raw_embeddings = _extract_embeddings(model, all_windows, device)
        embeddings, cluster_scan, cluster_model, cluster_labels = _build_pca_softmax_regime_outputs(
            embeddings=raw_embeddings,
            train_indices=prepared_inputs["window_split_indices"]["train"],
            target_cluster_count=int(target_cluster_count),
        )
    else:
        raw_embeddings = _extract_embeddings(model, all_windows, device)
        embeddings = raw_embeddings
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


def run_softmax_regime_transformer_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: Seq2SeqTransformerConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    config = model_config or Seq2SeqTransformerConfig(
        architecture="softmax_regime_transformer",
        embedding_dim=4,
    )
    config.architecture = "softmax_regime_transformer"
    config.embedding_dim = 4
    return _run_transformer_variant_experiment(
        experiment_name=experiment_name,
        architecture="softmax_regime_transformer",
        model_config=config,
        data_config=data_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )


def run_pca_softmax_regime_transformer_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: Seq2SeqTransformerConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    config = model_config or Seq2SeqTransformerConfig(
        architecture="pca_softmax_regime_transformer",
        embedding_dim=8,
    )
    config.architecture = "pca_softmax_regime_transformer"
    config.embedding_dim = 8
    return _run_transformer_variant_experiment(
        experiment_name=experiment_name,
        architecture="pca_softmax_regime_transformer",
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
    "run_softmax_regime_transformer_experiment",
    "run_pca_softmax_regime_transformer_experiment",
]
