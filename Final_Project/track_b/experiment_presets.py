from __future__ import annotations

from pathlib import Path

from advanced_transformer_variants import (
    ConvTransformerConfig,
    Seq2SeqTransformerConfig,
    run_cls_token_transformer_experiment,
    run_conv_transformer_experiment,
    run_mae_transformer_experiment,
    run_pca_softmax_regime_transformer_experiment,
    run_softmax_regime_transformer_experiment,
    run_tcn_transformer_experiment,
    run_transformer_autoencoder_experiment,
)
from paper_transformer_variants import (
    PatchTSTConfig,
    PathformerConfig,
    run_patchtst_experiment,
    run_pathformer_experiment,
)
from autoencoder_and_clustering_baselines import (
    ClassicalClusteringConfig,
    SequenceAutoencoderConfig,
    WindowVAEConfig,
    run_kmeans_baseline_experiment,
    run_lstm_autoencoder_experiment,
    run_pca_kmeans_experiment,
    run_rnn_autoencoder_experiment,
    run_vae_experiment,
)
from encoder_only_transformer import (
    ClusteringConfig,
    EncoderOnlyTransformerConfig,
    HMMReferenceConfig,
    SequenceDataConfig,
    TrainingConfig,
    run_encoder_only_experiment,
    run_market_tuple_encoder_only_experiment,
)
from rnn_baselines import (
    RecurrentBaselineConfig,
    run_pure_rnn_experiment,
    run_rnn_lstm_hybrid_experiment,
)


def build_architecture_runners() -> dict[str, object]:
    return {
        "encoder_only": run_encoder_only_experiment,
        "encoder_only_market_tuples": run_market_tuple_encoder_only_experiment,
        "patchtst": run_patchtst_experiment,
        "pathformer": run_pathformer_experiment,
        "cls_token_transformer": run_cls_token_transformer_experiment,
        "conv_transformer": run_conv_transformer_experiment,
        "tcn_transformer": run_tcn_transformer_experiment,
        "mae_transformer": run_mae_transformer_experiment,
        "softmax_regime_transformer": run_softmax_regime_transformer_experiment,
        "pca_softmax_regime_transformer": run_pca_softmax_regime_transformer_experiment,
        "pure_rnn": run_pure_rnn_experiment,
        "rnn_lstm_hybrid": run_rnn_lstm_hybrid_experiment,
        "rnn_autoencoder": run_rnn_autoencoder_experiment,
        "lstm_autoencoder": run_lstm_autoencoder_experiment,
        "transformer_autoencoder": run_transformer_autoencoder_experiment,
        "vae": run_vae_experiment,
        "kmeans_baseline": run_kmeans_baseline_experiment,
        "pca_kmeans_baseline": run_pca_kmeans_experiment,
    }


def build_default_data_config(data_dir: str | Path) -> SequenceDataConfig:
    return SequenceDataConfig(
        data_dir=Path(data_dir),
        market_filename="market_data_full_adjusted.csv",
        window_size=60,
        asset_return_columns=("SPY", "TLT", "GLD", "UUP", "HYG", "LQD"),
        include_vix_log_return=True,
        include_curve_slope_change=True,
        strict_feature_availability=False,
    )


def build_market_tuple_data_config(data_dir: str | Path) -> SequenceDataConfig:
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


def build_default_clustering_config(target_cluster_count: int = 4) -> ClusteringConfig:
    return ClusteringConfig(
        target_cluster_count=target_cluster_count,
        cluster_candidates=(target_cluster_count,),
        n_init=20,
        batch_size=256,
    )


def build_default_hmm_config(hmm_enabled: bool, target_cluster_count: int = 4) -> HMMReferenceConfig:
    return HMMReferenceConfig(
        enabled=hmm_enabled,
        state_candidates=(target_cluster_count,),
        covariance_type="diag",
        n_restarts=6,
        n_iter=500,
    )


def build_default_training_config(
    architecture_name: str,
    device: str = "auto",
) -> TrainingConfig:
    base_kwargs = dict(
        mask_ratio=0.20,
        batch_size=64,
        train_ratio=0.7,
        validation_ratio=0.2,
        test_ratio=0.1,
        early_stopping_patience=8,
        min_epochs=5,
        random_state=42,
        log_every=1,
        device=device,
    )

    presets = {
        "encoder_only": dict(num_epochs=80, learning_rate=1e-3, weight_decay=1e-4),
        "encoder_only_market_tuples": dict(num_epochs=80, learning_rate=1e-3, weight_decay=1e-4),
        "patchtst": dict(num_epochs=80, learning_rate=8e-4, weight_decay=1e-4),
        "pathformer": dict(num_epochs=80, learning_rate=8e-4, weight_decay=1e-4),
        "cls_token_transformer": dict(num_epochs=80, learning_rate=1e-3, weight_decay=1e-4),
        "conv_transformer": dict(num_epochs=60, learning_rate=8e-4, weight_decay=1e-4),
        "tcn_transformer": dict(num_epochs=60, learning_rate=8e-4, weight_decay=1e-4),
        "mae_transformer": dict(num_epochs=60, learning_rate=8e-4, weight_decay=1e-4),
        "softmax_regime_transformer": dict(num_epochs=60, learning_rate=8e-4, weight_decay=1e-4),
        "pca_softmax_regime_transformer": dict(num_epochs=60, learning_rate=8e-4, weight_decay=1e-4),
        "pure_rnn": dict(num_epochs=80, learning_rate=1e-3, weight_decay=1e-4),
        "rnn_lstm_hybrid": dict(num_epochs=80, learning_rate=1e-3, weight_decay=1e-4),
        "rnn_autoencoder": dict(num_epochs=60, learning_rate=1e-3, weight_decay=1e-4),
        "lstm_autoencoder": dict(num_epochs=60, learning_rate=1e-3, weight_decay=1e-4),
        "transformer_autoencoder": dict(num_epochs=60, learning_rate=8e-4, weight_decay=1e-4),
        "vae": dict(num_epochs=60, learning_rate=1e-3, weight_decay=1e-5),
        "kmeans_baseline": dict(num_epochs=1, learning_rate=1e-3, weight_decay=0.0),
        "pca_kmeans_baseline": dict(num_epochs=1, learning_rate=1e-3, weight_decay=0.0),
    }
    return TrainingConfig(**base_kwargs, **presets[architecture_name])


def build_default_model_config(architecture_name: str):
    configs = {
        "encoder_only": EncoderOnlyTransformerConfig(
            d_model=64,
            embedding_dim=64,
            n_heads=4,
            num_layers=2,
            dropout=0.10,
            feedforward_multiplier=2.0,
            pooling="attention",
            use_mask_embedding=True,
        ),
        "encoder_only_market_tuples": EncoderOnlyTransformerConfig(
            d_model=128,
            embedding_dim=64,
            n_heads=8,
            num_layers=2,
            dropout=0.10,
            feedforward_multiplier=2.0,
            pooling="attention",
            use_mask_embedding=True,
        ),
        "patchtst": PatchTSTConfig(
            architecture="patchtst",
            patch_len=6,
            patch_stride=3,
            d_model=64,
            embedding_dim=64,
            n_heads=4,
            num_layers=2,
            dropout=0.10,
            feedforward_multiplier=2.0,
            pooling="attention",
            channel_pooling="attention",
            use_mask_embedding=True,
        ),
        "pathformer": PathformerConfig(
            architecture="pathformer",
            patch_sizes=(4, 8, 12),
            d_model=64,
            embedding_dim=64,
            n_heads=4,
            num_layers=2,
            dropout=0.10,
            feedforward_multiplier=2.0,
            pooling="attention",
            use_mask_embedding=True,
            router_hidden_dim=64,
        ),
        "cls_token_transformer": EncoderOnlyTransformerConfig(
            d_model=64,
            embedding_dim=64,
            n_heads=4,
            num_layers=2,
            dropout=0.10,
            feedforward_multiplier=2.0,
            pooling="cls",
            use_mask_embedding=True,
        ),
        "conv_transformer": ConvTransformerConfig(
            architecture="conv_transformer",
            conv_channels=64,
            d_model=64,
            embedding_dim=64,
            n_heads=4,
            num_transformer_layers=2,
            num_conv_layers=2,
            kernel_size=3,
            dilation_base=2,
            dropout=0.10,
            feedforward_multiplier=2.0,
            pooling="attention",
            use_mask_embedding=True,
        ),
        "tcn_transformer": ConvTransformerConfig(
            architecture="tcn_transformer",
            conv_channels=64,
            d_model=64,
            embedding_dim=64,
            n_heads=4,
            num_transformer_layers=2,
            num_conv_layers=3,
            kernel_size=3,
            dilation_base=2,
            dropout=0.10,
            feedforward_multiplier=2.0,
            pooling="attention",
            use_mask_embedding=True,
        ),
        "mae_transformer": Seq2SeqTransformerConfig(
            architecture="mae_transformer",
            d_model=64,
            embedding_dim=64,
            n_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=1,
            dropout=0.10,
            feedforward_multiplier=2.0,
            pooling="mean",
            use_mask_embedding=True,
        ),
        "softmax_regime_transformer": Seq2SeqTransformerConfig(
            architecture="softmax_regime_transformer",
            d_model=64,
            embedding_dim=4,
            n_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=1,
            dropout=0.10,
            feedforward_multiplier=2.0,
            pooling="mean",
            use_mask_embedding=True,
        ),
        "pca_softmax_regime_transformer": Seq2SeqTransformerConfig(
            architecture="pca_softmax_regime_transformer",
            d_model=64,
            embedding_dim=8,
            n_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=1,
            dropout=0.10,
            feedforward_multiplier=2.0,
            pooling="mean",
            use_mask_embedding=True,
        ),
        "pure_rnn": RecurrentBaselineConfig(
            architecture="pure_rnn",
            input_projection_dim=32,
            rnn_hidden_dim=64,
            rnn_num_layers=2,
            lstm_hidden_dim=64,
            lstm_num_layers=1,
            embedding_dim=64,
            dropout=0.10,
            pooling="attention",
            nonlinearity="tanh",
            bidirectional=False,
            use_mask_embedding=True,
        ),
        "rnn_lstm_hybrid": RecurrentBaselineConfig(
            architecture="rnn_lstm_hybrid",
            input_projection_dim=32,
            rnn_hidden_dim=64,
            rnn_num_layers=1,
            lstm_hidden_dim=64,
            lstm_num_layers=1,
            embedding_dim=64,
            dropout=0.10,
            pooling="attention",
            nonlinearity="tanh",
            bidirectional=False,
            use_mask_embedding=True,
        ),
        "rnn_autoencoder": SequenceAutoencoderConfig(
            architecture="rnn_autoencoder",
            input_projection_dim=32,
            hidden_dim=64,
            num_layers=2,
            embedding_dim=32,
            decoder_hidden_dim=64,
            dropout=0.10,
            nonlinearity="tanh",
            bidirectional=False,
        ),
        "lstm_autoencoder": SequenceAutoencoderConfig(
            architecture="lstm_autoencoder",
            input_projection_dim=32,
            hidden_dim=64,
            num_layers=2,
            embedding_dim=32,
            decoder_hidden_dim=64,
            dropout=0.10,
            nonlinearity="tanh",
            bidirectional=False,
        ),
        "transformer_autoencoder": Seq2SeqTransformerConfig(
            architecture="transformer_autoencoder",
            d_model=64,
            embedding_dim=64,
            n_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=1,
            dropout=0.10,
            feedforward_multiplier=2.0,
            pooling="mean",
            use_mask_embedding=True,
        ),
        "vae": WindowVAEConfig(
            architecture="vae",
            hidden_dim=256,
            latent_dim=32,
            beta=1e-3,
            dropout=0.10,
        ),
        "kmeans_baseline": ClassicalClusteringConfig(
            architecture="kmeans_baseline",
            pca_components=16,
        ),
        "pca_kmeans_baseline": ClassicalClusteringConfig(
            architecture="pca_kmeans_baseline",
            pca_components=16,
        ),
    }
    return configs[architecture_name]


def build_default_experiment_setups(
    data_dir: str | Path,
    hmm_enabled: bool,
    target_cluster_count: int = 4,
    device: str = "auto",
) -> list[dict]:
    architecture_names = [
        "encoder_only",
        "encoder_only_market_tuples",
        "patchtst",
        "pathformer",
        "cls_token_transformer",
        "conv_transformer",
        "tcn_transformer",
        "mae_transformer",
        "softmax_regime_transformer",
        "pca_softmax_regime_transformer",
        "pure_rnn",
        "rnn_lstm_hybrid",
        "rnn_autoencoder",
        "lstm_autoencoder",
        "transformer_autoencoder",
        "vae",
        "kmeans_baseline",
        "pca_kmeans_baseline",
    ]
    return [
        {
            "name": architecture_name,
            "architecture": architecture_name,
            "data_config": (
                build_market_tuple_data_config(data_dir)
                if architecture_name == "encoder_only_market_tuples"
                else build_default_data_config(data_dir)
            ),
            "model_config": build_default_model_config(architecture_name),
            "training_config": build_default_training_config(architecture_name, device=device),
            "clustering_config": build_default_clustering_config(target_cluster_count=target_cluster_count),
            "hmm_config": build_default_hmm_config(hmm_enabled=hmm_enabled, target_cluster_count=target_cluster_count),
        }
        for architecture_name in architecture_names
    ]


__all__ = [
    "build_architecture_runners",
    "build_default_data_config",
    "build_market_tuple_data_config",
    "build_default_model_config",
    "build_default_training_config",
    "build_default_clustering_config",
    "build_default_hmm_config",
    "build_default_experiment_setups",
]
