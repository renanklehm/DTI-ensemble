from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class DataConfig:
    path: str = "data/1_ready/df.parquet"
    gsdti_source: str = "GSDTI"
    dcgat_source: str = "DCGAT-DTI"
    smiles_column: str = "smiles"
    sequence_column: str = "sequence"
    target_column: str = "real_value"
    source_column: str = "source"
    label_column: str = "label"
    probability_column: str = "probability"


@dataclass
class SplitConfig:
    strategy: str = "stratified_random"
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42


@dataclass
class EncoderConfig:
    model_name: str
    max_length: int
    batch_size: int = 32
    pooling: str = "mean"
    dtype: str = "float32"
    trust_remote_code: bool = False


@dataclass
class ModelConfig:
    projection_dim: int = 256
    base_projection_dim: int = 64
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 64])
    dropout: float = 0.25


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    warmup_fraction: float = 0.05
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    pin_memory: bool = True
    mixed_precision: bool = True
    patience: int = 7
    min_delta: float = 1.0e-4
    monitor: str = "auprc"
    monitor_mode: str = "max"
    class_weight: str = "balanced"
    threshold_metric: str = "f1"
    seed: int = 42
    deterministic: bool = False


@dataclass
class RuntimeConfig:
    output_dir: str = "artifacts/ensemble"
    cache_dir: str = "artifacts/embedding_cache"
    device: str = "auto"
    log_level: str = "INFO"


@dataclass
class EnsembleConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    smiles_encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(
            model_name="DeepChem/ChemBERTa-77M-MLM", max_length=256, batch_size=64
        )
    )
    protein_encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(
            model_name="facebook/esm2_t6_8M_UR50D", max_length=1024, batch_size=8
        )
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _update_dataclass(instance: T, values: dict[str, Any], prefix: str = "") -> T:
    known = {item.name for item in fields(instance)}
    unknown = set(values) - known
    if unknown:
        location = prefix or "config"
        raise ValueError(f"Unknown {location} keys: {sorted(unknown)}")
    for name, value in values.items():
        current = getattr(instance, name)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{prefix}{name} must be a mapping")
            _update_dataclass(current, value, f"{prefix}{name}.")
        else:
            setattr(instance, name, value)
    return instance


def load_config(path: str | Path) -> EnsembleConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("The YAML root must be a mapping")
    config = _update_dataclass(EnsembleConfig(), raw)
    validate_config(config)
    return config


def validate_config(config: EnsembleConfig) -> None:
    fractions = config.split.validation_fraction + config.split.test_fraction
    if not 0.0 < fractions < 1.0:
        raise ValueError("validation_fraction + test_fraction must be between 0 and 1")
    if config.split.strategy not in {"stratified_random", "drug_cold", "target_cold"}:
        raise ValueError("split.strategy must be stratified_random, drug_cold, or target_cold")
    if config.training.monitor_mode not in {"min", "max"}:
        raise ValueError("training.monitor_mode must be min or max")
    if config.training.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if config.training.class_weight not in {"balanced", "none"}:
        raise ValueError("training.class_weight must be balanced or none")
    if config.training.monitor not in {
        "loss", "auroc", "auprc", "accuracy", "balanced_accuracy", "f1",
        "precision", "recall", "mcc", "brier",
    }:
        raise ValueError("training.monitor is not a supported validation metric")
    if config.training.threshold_metric not in {"f1", "balanced_accuracy", "mcc"}:
        raise ValueError("threshold_metric must be f1, balanced_accuracy, or mcc")
    if config.training.batch_size < 1 or config.training.epochs < 1:
        raise ValueError("training batch_size and epochs must be at least 1")
    for encoder in (config.smiles_encoder, config.protein_encoder):
        if encoder.pooling not in {"mean", "cls"}:
            raise ValueError("encoder pooling must be mean or cls")
        if encoder.dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError("encoder dtype must be float16, float32, or bfloat16")
        if encoder.batch_size < 1 or encoder.max_length < 3:
            raise ValueError("encoder batch_size must be positive and max_length at least 3")


def save_config(config: EnsembleConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
