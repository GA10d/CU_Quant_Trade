from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from advanced_transformer_variants import _extract_embeddings, _train_masked_model
from encoder_only_transformer import (
    ClusteringConfig,
    HMMReferenceConfig,
    SequenceDataConfig,
    SinusoidalPositionalEncoding,
    TrainingConfig,
    cluster_embeddings,
    set_seed,
)
from experiment_utils import (
    build_experiment_result,
    prepare_hmm_reference_for_experiment,
    prepare_sequence_experiment_inputs,
    resolve_target_cluster_count,
    summarize_experiment_outputs,
)


@dataclass
class PatchTSTConfig:
    architecture: str = "patchtst"
    patch_len: int = 6
    patch_stride: int = 3
    d_model: int = 64
    embedding_dim: int = 64
    n_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.10
    feedforward_multiplier: float = 2.0
    pooling: str = "attention"
    channel_pooling: str = "attention"
    use_mask_embedding: bool = True


@dataclass
class PathformerConfig:
    architecture: str = "pathformer"
    patch_sizes: tuple[int, ...] = (4, 8, 12)
    d_model: int = 64
    embedding_dim: int = 64
    n_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.10
    feedforward_multiplier: float = 2.0
    pooling: str = "attention"
    use_mask_embedding: bool = True
    router_hidden_dim: int = 64


def _compute_patch_starts(seq_len: int, patch_len: int, stride: int) -> list[int]:
    if patch_len <= 0 or stride <= 0:
        raise ValueError("patch_len and stride must be positive integers.")
    if seq_len < patch_len:
        raise ValueError(f"Sequence length {seq_len} is smaller than patch_len={patch_len}.")

    starts = list(range(0, seq_len - patch_len + 1, stride))
    if starts[-1] != seq_len - patch_len:
        starts.append(seq_len - patch_len)
    return starts


class PatchTSTBackbone(nn.Module):
    def __init__(self, input_dim: int, window_size: int, model_config: PatchTSTConfig):
        super().__init__()
        if model_config.pooling not in {"attention", "mean", "last"}:
            raise ValueError("pooling must be one of: attention, mean, last")
        if model_config.channel_pooling not in {"attention", "mean"}:
            raise ValueError("channel_pooling must be one of: attention, mean")
        if model_config.d_model % model_config.n_heads != 0:
            raise ValueError(
                f"d_model={model_config.d_model} must be divisible by n_heads={model_config.n_heads}."
            )

        self.input_dim = input_dim
        self.window_size = window_size
        self.patch_len = model_config.patch_len
        self.patch_stride = model_config.patch_stride
        self.patch_starts = _compute_patch_starts(window_size, self.patch_len, self.patch_stride)
        self.pooling = model_config.pooling
        self.channel_pooling = model_config.channel_pooling
        self.use_mask_embedding = model_config.use_mask_embedding

        self.input_norm = nn.LayerNorm(input_dim)
        self.patch_embed = nn.Linear(self.patch_len, model_config.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model_config.d_model))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        self.pos_encoder = SinusoidalPositionalEncoding(model_config.d_model, max_len=len(self.patch_starts))
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
            num_layers=model_config.num_layers,
            norm=nn.LayerNorm(model_config.d_model),
        )

        self.patch_reconstruct = nn.Sequential(
            nn.LayerNorm(model_config.d_model),
            nn.Linear(model_config.d_model, model_config.d_model),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.d_model, self.patch_len),
        )
        self.patch_pool_score = nn.Linear(model_config.d_model, 1) if self.pooling == "attention" else None
        self.channel_pool_score = (
            nn.Linear(model_config.d_model, 1) if self.channel_pooling == "attention" else None
        )
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(model_config.d_model),
            nn.Linear(model_config.d_model, model_config.d_model),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.d_model, model_config.embedding_dim),
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        patches = [x[:, start : start + self.patch_len, :].permute(0, 2, 1) for start in self.patch_starts]
        return torch.stack(patches, dim=2)

    def _patchify_mask(self, mask: torch.Tensor) -> torch.Tensor:
        patch_masks = [
            mask[:, start : start + self.patch_len, :].permute(0, 2, 1).any(dim=-1)
            for start in self.patch_starts
        ]
        return torch.stack(patch_masks, dim=2)

    def _unpatchify(self, patch_values: torch.Tensor) -> torch.Tensor:
        batch_size, input_dim, n_patches, _ = patch_values.shape
        reconstruction = patch_values.new_zeros(batch_size, input_dim, self.window_size)
        counts = patch_values.new_zeros(batch_size, input_dim, self.window_size)

        for patch_idx, start in enumerate(self.patch_starts):
            end = start + self.patch_len
            reconstruction[:, :, start:end] += patch_values[:, :, patch_idx, :]
            counts[:, :, start:end] += 1.0

        reconstruction = reconstruction / counts.clamp_min(1.0)
        return reconstruction.permute(0, 2, 1)

    def _pool_tokens(self, hidden: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        if token_mask is not None:
            valid_mask = (~token_mask).float().unsqueeze(-1)
        else:
            valid_mask = None

        if self.pooling == "attention":
            assert self.patch_pool_score is not None
            scores = self.patch_pool_score(hidden)
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

    def _pool_channels(self, channel_hidden: torch.Tensor) -> torch.Tensor:
        if self.channel_pooling == "mean":
            return channel_hidden.mean(dim=1)

        assert self.channel_pool_score is not None
        scores = self.channel_pool_score(channel_hidden)
        weights = torch.softmax(scores, dim=1)
        return (channel_hidden * weights).sum(dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_x = self.input_norm(x)
        patches = self._patchify(normalized_x)
        batch_size, input_dim, n_patches, _ = patches.shape

        patch_tokens = self.patch_embed(patches.reshape(batch_size * input_dim, n_patches, self.patch_len))
        patch_mask = None
        if mask is not None:
            patch_mask = self._patchify_mask(mask).reshape(batch_size * input_dim, n_patches)
            if patch_mask.all(dim=1).any():
                patch_mask = patch_mask.clone()
                fully_masked = patch_mask.all(dim=1)
                patch_mask[fully_masked, 0] = False
            if self.use_mask_embedding:
                patch_tokens = patch_tokens + self.mask_token * patch_mask.unsqueeze(-1).float()

        patch_tokens = self.pos_dropout(self.pos_encoder(patch_tokens))
        hidden = self.encoder(patch_tokens, src_key_padding_mask=patch_mask)

        patch_reconstruction = self.patch_reconstruct(hidden).reshape(
            batch_size,
            input_dim,
            n_patches,
            self.patch_len,
        )
        reconstruction = self._unpatchify(patch_reconstruction)

        channel_hidden = self._pool_tokens(hidden, token_mask=patch_mask).reshape(batch_size, input_dim, -1)
        embedding = self.embedding_head(self._pool_channels(channel_hidden))
        return reconstruction, embedding


class _PathformerScaleBlock(nn.Module):
    def __init__(self, input_dim: int, window_size: int, patch_size: int, model_config: PathformerConfig):
        super().__init__()
        stride = max(1, patch_size // 2)
        self.input_dim = input_dim
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_starts = _compute_patch_starts(window_size, patch_size, stride)
        self.pooling = model_config.pooling
        self.use_mask_embedding = model_config.use_mask_embedding

        self.patch_embed = nn.Linear(input_dim * patch_size, model_config.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model_config.d_model))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        self.pos_encoder = SinusoidalPositionalEncoding(model_config.d_model, max_len=len(self.patch_starts))
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
            num_layers=model_config.num_layers,
            norm=nn.LayerNorm(model_config.d_model),
        )
        self.local_mixer = nn.Sequential(
            nn.Conv1d(model_config.d_model, model_config.d_model, kernel_size=3, padding=1, groups=model_config.n_heads),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Conv1d(model_config.d_model, model_config.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
        )
        self.patch_reconstruct = nn.Sequential(
            nn.LayerNorm(model_config.d_model),
            nn.Linear(model_config.d_model, model_config.d_model),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.d_model, input_dim * patch_size),
        )
        self.token_pool_score = nn.Linear(model_config.d_model, 1) if model_config.pooling == "attention" else None

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        patches = [x[:, start : start + self.patch_size, :].reshape(x.size(0), -1) for start in self.patch_starts]
        return torch.stack(patches, dim=1)

    def _patchify_mask(self, mask: torch.Tensor) -> torch.Tensor:
        patch_masks = [mask[:, start : start + self.patch_size, :].any(dim=(1, 2)) for start in self.patch_starts]
        return torch.stack(patch_masks, dim=1)

    def _unpatchify(self, patch_values: torch.Tensor) -> torch.Tensor:
        batch_size, n_patches, _ = patch_values.shape
        reconstruction = patch_values.new_zeros(batch_size, self.window_size, self.input_dim)
        counts = patch_values.new_zeros(batch_size, self.window_size, self.input_dim)

        reshaped = patch_values.reshape(batch_size, n_patches, self.patch_size, self.input_dim)
        for patch_idx, start in enumerate(self.patch_starts):
            end = start + self.patch_size
            reconstruction[:, start:end, :] += reshaped[:, patch_idx, :, :]
            counts[:, start:end, :] += 1.0

        return reconstruction / counts.clamp_min(1.0)

    def _pool_tokens(self, hidden: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        if token_mask is not None:
            valid_mask = (~token_mask).float().unsqueeze(-1)
        else:
            valid_mask = None

        if self.pooling == "attention":
            assert self.token_pool_score is not None
            scores = self.token_pool_score(hidden)
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
        patch_values = self._patchify(x)
        token_mask = self._patchify_mask(mask) if mask is not None else None
        if token_mask is not None and token_mask.all(dim=1).any():
            token_mask = token_mask.clone()
            fully_masked = token_mask.all(dim=1)
            token_mask[fully_masked, 0] = False

        hidden = self.patch_embed(patch_values)
        if token_mask is not None and self.use_mask_embedding:
            hidden = hidden + self.mask_token * token_mask.unsqueeze(-1).float()

        hidden = self.pos_dropout(self.pos_encoder(hidden))
        global_hidden = self.encoder(hidden, src_key_padding_mask=token_mask)
        local_hidden = self.local_mixer(hidden.transpose(1, 2)).transpose(1, 2)
        fused_hidden = global_hidden + local_hidden

        patch_reconstruction = self.patch_reconstruct(fused_hidden)
        reconstruction = self._unpatchify(patch_reconstruction)
        embedding = self._pool_tokens(fused_hidden, token_mask=token_mask)
        return reconstruction, embedding


class PathformerBackbone(nn.Module):
    def __init__(self, input_dim: int, window_size: int, model_config: PathformerConfig):
        super().__init__()
        if model_config.pooling not in {"attention", "mean", "last"}:
            raise ValueError("pooling must be one of: attention, mean, last")
        if model_config.d_model % model_config.n_heads != 0:
            raise ValueError(
                f"d_model={model_config.d_model} must be divisible by n_heads={model_config.n_heads}."
            )

        self.input_dim = input_dim
        self.window_size = window_size
        self.input_norm = nn.LayerNorm(input_dim)
        self.scale_blocks = nn.ModuleList(
            [
                _PathformerScaleBlock(
                    input_dim=input_dim,
                    window_size=window_size,
                    patch_size=patch_size,
                    model_config=model_config,
                )
                for patch_size in model_config.patch_sizes
            ]
        )
        self.router = nn.Sequential(
            nn.Linear(input_dim * 2, model_config.router_hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.router_hidden_dim, len(model_config.patch_sizes)),
        )
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(model_config.d_model),
            nn.Linear(model_config.d_model, model_config.d_model),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.d_model, model_config.embedding_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_x = self.input_norm(x)
        input_stats = torch.cat(
            [
                normalized_x.mean(dim=1),
                normalized_x.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        router_weights = torch.softmax(self.router(input_stats), dim=-1)

        scale_recons = []
        scale_embeddings = []
        for scale_idx, block in enumerate(self.scale_blocks):
            scale_recon, scale_embedding = block(normalized_x, mask=mask)
            scale_recons.append(scale_recon)
            scale_embeddings.append(scale_embedding)

        stacked_recons = torch.stack(scale_recons, dim=1)
        stacked_embeddings = torch.stack(scale_embeddings, dim=1)
        recon_weights = router_weights.unsqueeze(-1).unsqueeze(-1)
        embed_weights = router_weights.unsqueeze(-1)

        reconstruction = (stacked_recons * recon_weights).sum(dim=1)
        embedding = self.embedding_head((stacked_embeddings * embed_weights).sum(dim=1))
        return reconstruction, embedding


def _run_paper_transformer_experiment(
    experiment_name: str,
    architecture: str,
    model: nn.Module,
    data_config: SequenceDataConfig | None,
    training_config: TrainingConfig | None,
    clustering_config: ClusteringConfig | None,
    hmm_config: HMMReferenceConfig | None,
    model_config: PatchTSTConfig | PathformerConfig,
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

    model, history_df, all_windows, device = _train_masked_model(
        windows=prepared_inputs["windows"],
        model=model,
        training_config=training_config,
        log_prefix=architecture,
        split_indices=prepared_inputs["window_split_indices"],
    )
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


def run_patchtst_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: PatchTSTConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    model_config = model_config or PatchTSTConfig()
    data_config = data_config or SequenceDataConfig()
    prepared_inputs = prepare_sequence_experiment_inputs(data_config, training_config=training_config)
    model = PatchTSTBackbone(
        input_dim=prepared_inputs["windows"].shape[2],
        window_size=data_config.window_size,
        model_config=model_config,
    )
    return _run_paper_transformer_experiment(
        experiment_name=experiment_name,
        architecture="patchtst",
        model=model,
        data_config=data_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
        model_config=model_config,
    )


def run_pathformer_experiment(
    experiment_name: str,
    data_config: SequenceDataConfig | None = None,
    model_config: PathformerConfig | None = None,
    training_config: TrainingConfig | None = None,
    clustering_config: ClusteringConfig | None = None,
    hmm_config: HMMReferenceConfig | None = None,
) -> dict:
    model_config = model_config or PathformerConfig()
    data_config = data_config or SequenceDataConfig()
    prepared_inputs = prepare_sequence_experiment_inputs(data_config, training_config=training_config)
    model = PathformerBackbone(
        input_dim=prepared_inputs["windows"].shape[2],
        window_size=data_config.window_size,
        model_config=model_config,
    )
    return _run_paper_transformer_experiment(
        experiment_name=experiment_name,
        architecture="pathformer",
        model=model,
        data_config=data_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
        model_config=model_config,
    )


__all__ = [
    "PatchTSTConfig",
    "PathformerConfig",
    "PatchTSTBackbone",
    "PathformerBackbone",
    "run_patchtst_experiment",
    "run_pathformer_experiment",
]
