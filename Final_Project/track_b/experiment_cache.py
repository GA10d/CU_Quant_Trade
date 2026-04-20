from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from encoder_only_transformer import SequenceDataConfig
from experiment_utils import prepare_sequence_experiment_inputs


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = MODULE_DIR / "models"


def _normalize_for_json(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_for_json(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _data_fingerprint(data_config: SequenceDataConfig) -> dict[str, dict[str, Any]]:
    data_dir = Path(data_config.data_dir)
    fingerprints: dict[str, dict[str, Any]] = {}
    for attr_name in ("market_filename", "macro_filename", "hmm_features_filename"):
        filename = getattr(data_config, attr_name, None)
        if not filename:
            continue
        path = data_dir / filename
        if path.exists():
            stat = path.stat()
            fingerprints[attr_name] = {
                "path": str(path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        else:
            fingerprints[attr_name] = {"path": str(path.resolve()), "missing": True}
    return fingerprints


def build_experiment_cache_key(
    experiment_name: str,
    data_config: Any,
    model_config: Any,
    training_config: Any,
    clustering_config: Any,
    hmm_config: Any,
) -> str:
    payload = {
        "experiment_name": experiment_name,
        "data_config": _normalize_for_json(data_config),
        "model_config": _normalize_for_json(model_config),
        "training_config": _normalize_for_json(training_config),
        "clustering_config": _normalize_for_json(clustering_config),
        "hmm_config": _normalize_for_json(hmm_config),
        "data_fingerprint": _data_fingerprint(data_config),
    }
    payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload_text.encode("utf-8")).hexdigest()[:16]


def _safe_experiment_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def get_experiment_cache_dir(
    models_dir: str | Path,
    experiment_name: str,
    cache_key: str,
) -> Path:
    models_path = Path(models_dir)
    safe_name = _safe_experiment_name(experiment_name)
    return models_path / f"{safe_name}__{cache_key}"


def _write_optional_dataframe(frame: pd.DataFrame | None, path: Path, index: bool = False) -> None:
    if frame is None:
        return
    frame.to_csv(path, index=index)


def _read_optional_dataframe(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_experiment_result_to_cache(
    result: dict,
    cache_dir: str | Path,
) -> Path:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    np.save(cache_path / "embeddings.npy", result["embeddings"])
    np.save(cache_path / "cluster_labels.npy", result["cluster_labels"])

    pd.Index(result["window_end_dates"], name="Date").to_series().to_csv(cache_path / "window_end_dates.csv", index=False)
    _write_optional_dataframe(result.get("history_df"), cache_path / "history_df.csv", index=False)
    _write_optional_dataframe(result.get("cluster_scan"), cache_path / "cluster_scan.csv", index=False)
    _write_optional_dataframe(result.get("cluster_summary"), cache_path / "cluster_summary.csv", index=False)
    _write_optional_dataframe(result.get("comparison_table"), cache_path / "comparison_table.csv", index=True)

    hmm_results = result.get("hmm_results")
    if hmm_results is not None and "hmm_reference" in hmm_results:
        hmm_results["hmm_reference"].to_csv(cache_path / "hmm_reference.csv", index=True)

    model = result.get("model")
    model_state_path = None
    if isinstance(model, nn.Module):
        model_state_path = cache_path / "model_state.pt"
        torch.save(model.state_dict(), model_state_path)

    metadata = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": result.get("experiment_name"),
        "summary": _normalize_for_json(result.get("summary", {})),
        "model_state_path": str(model_state_path) if model_state_path is not None else None,
        "device": result.get("device"),
    }
    (cache_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cache_path


def load_experiment_result_from_cache(
    cache_dir: str | Path,
    experiment_name: str,
    data_config: Any,
    model_config: Any,
    training_config: Any,
    clustering_config: Any,
    hmm_config: Any,
) -> dict:
    cache_path = Path(cache_dir)
    metadata = json.loads((cache_path / "metadata.json").read_text(encoding="utf-8"))

    prepared_inputs = prepare_sequence_experiment_inputs(data_config)
    embeddings = np.load(cache_path / "embeddings.npy")
    cluster_labels = np.load(cache_path / "cluster_labels.npy")
    stored_window_dates = pd.read_csv(cache_path / "window_end_dates.csv")["Date"]
    window_end_dates = pd.Index(pd.to_datetime(stored_window_dates), name="Date")

    if len(window_end_dates) != len(prepared_inputs["window_end_dates"]):
        raise ValueError(
            "Cached window_end_dates length does not match the current prepared inputs. "
            "The underlying dataset or preprocessing configuration likely changed."
        )

    history_df = _read_optional_dataframe(cache_path / "history_df.csv")
    cluster_scan = _read_optional_dataframe(cache_path / "cluster_scan.csv")
    cluster_summary = _read_optional_dataframe(cache_path / "cluster_summary.csv")
    comparison_table = _read_optional_dataframe(cache_path / "comparison_table.csv", index_col=0)

    hmm_reference_path = cache_path / "hmm_reference.csv"
    hmm_results = None
    if hmm_reference_path.exists():
        hmm_reference = pd.read_csv(hmm_reference_path, index_col=0, parse_dates=True)
        hmm_reference.index.name = "Date"
        hmm_results = {"hmm_reference": hmm_reference}

    return {
        "experiment_name": experiment_name,
        "summary": metadata["summary"],
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
        "window_end_dates": window_end_dates,
        "model": None,
        "model_state_path": metadata.get("model_state_path"),
        "history_df": history_df if history_df is not None else pd.DataFrame(),
        "embeddings": embeddings,
        "cluster_scan": cluster_scan if cluster_scan is not None else pd.DataFrame(),
        "cluster_model": None,
        "cluster_labels": cluster_labels,
        "cluster_summary": cluster_summary if cluster_summary is not None else pd.DataFrame(),
        "comparison_table": comparison_table,
        "hmm_results": hmm_results,
        "device": "cached",
        "cache_dir": str(cache_path),
    }


def run_experiment_with_cache(
    runner: Callable[..., dict],
    experiment_name: str,
    data_config: Any,
    model_config: Any,
    training_config: Any,
    clustering_config: Any,
    hmm_config: Any,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    verbose: bool = True,
) -> dict:
    cache_key = build_experiment_cache_key(
        experiment_name=experiment_name,
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )
    cache_dir = get_experiment_cache_dir(models_dir=models_dir, experiment_name=experiment_name, cache_key=cache_key)
    metadata_path = cache_dir / "metadata.json"

    if metadata_path.exists():
        if verbose:
            print(f"[cache] loaded {experiment_name} from {cache_dir}")
        return load_experiment_result_from_cache(
            cache_dir=cache_dir,
            experiment_name=experiment_name,
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            clustering_config=clustering_config,
            hmm_config=hmm_config,
        )

    if verbose:
        print(f"[cache] training {experiment_name} and saving to {cache_dir}")
    result = runner(
        experiment_name=experiment_name,
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        clustering_config=clustering_config,
        hmm_config=hmm_config,
    )
    save_experiment_result_to_cache(result=result, cache_dir=cache_dir)
    result["cache_dir"] = str(cache_dir)
    return result


__all__ = [
    "DEFAULT_MODELS_DIR",
    "build_experiment_cache_key",
    "get_experiment_cache_dir",
    "save_experiment_result_to_cache",
    "load_experiment_result_from_cache",
    "run_experiment_with_cache",
]
