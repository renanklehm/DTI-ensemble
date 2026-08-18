from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import EnsembleConfig, load_config, save_config
from .data import (
    BASE_FEATURE_COLUMNS,
    EnsembleDataset,
    build_paired_frame,
    create_splits,
    read_table,
)
from .embeddings import load_or_compute_embeddings, resolve_device
from .logging_utils import configure_logging
from .metrics import binary_metrics
from .model import EnsembleModel
from .trainer import Trainer, make_loader, predict, seed_everything, write_metrics

LOGGER = logging.getLogger(__name__)


def prepare_frame(config: EnsembleConfig) -> pd.DataFrame:
    LOGGER.info("Reading source data from %s", config.data.path)
    raw = read_table(config.data.path)
    LOGGER.info("Read %d long-format rows", len(raw))
    return create_splits(build_paired_frame(raw, config.data), config.split)


def _new_run_dir(root: str | Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = Path(root) / timestamp
    suffix = 1
    while candidate.exists():
        candidate = Path(root) / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def train_pipeline(config: EnsembleConfig, resume: str | None = None) -> Path:
    run_dir = Path(resume).resolve().parent if resume else _new_run_dir(config.runtime.output_dir)
    configure_logging(run_dir, config.runtime.log_level)
    save_config(config, run_dir / "resolved_config.yaml")
    seed_everything(config.training.seed, config.training.deterministic)
    device = resolve_device(config.runtime.device)
    LOGGER.info("Training run: %s", run_dir)
    LOGGER.info("Torch %s | device=%s | CUDA=%s", torch.__version__, device, torch.cuda.is_available())

    split_path = run_dir / "paired_splits.parquet"
    if resume and split_path.exists():
        frame = pd.read_parquet(split_path)
        LOGGER.info("Reusing persisted split assignments from %s", split_path)
    else:
        frame = prepare_frame(config)
        frame.to_parquet(split_path, index=False)
        LOGGER.info("Saved aligned split table to %s", split_path)

    LOGGER.info(
        "Entities: %d unique SMILES, %d unique sequences | max lengths: %d, %d characters",
        frame["smiles"].nunique(),
        frame["sequence"].nunique(),
        int(frame["smiles"].str.len().max()),
        int(frame["sequence"].str.len().max()),
    )

    smiles_embeddings = load_or_compute_embeddings(
        frame["smiles"].tolist(),
        "smiles",
        config.smiles_encoder,
        config.runtime.cache_dir,
        device,
    )
    sequence_embeddings = load_or_compute_embeddings(
        frame["sequence"].tolist(),
        "sequence",
        config.protein_encoder,
        config.runtime.cache_dir,
        device,
    )
    subsets = {name: group.reset_index(drop=True) for name, group in frame.groupby("split")}
    datasets = {
        name: EnsembleDataset(group, smiles_embeddings, sequence_embeddings)
        for name, group in subsets.items()
    }
    loaders = {
        name: make_loader(dataset, config.training, shuffle=name == "train")
        for name, dataset in datasets.items()
    }
    train_features = datasets["train"].base_features
    base_mean = train_features.mean(dim=0)
    base_std = train_features.std(dim=0).clamp_min(1.0e-6)
    model = EnsembleModel(
        smiles_dim=smiles_embeddings.values.shape[1],
        sequence_dim=sequence_embeddings.values.shape[1],
        config=config.model,
        base_mean=base_mean,
        base_std=base_std,
    )
    positives = float(datasets["train"].targets.sum())
    negatives = len(datasets["train"]) - positives
    positive_weight = negatives / positives if config.training.class_weight == "balanced" else 1.0
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    LOGGER.info(
        "Fusion model has %s trainable parameters; positive weight=%.4f",
        f"{trainable_parameters:,}",
        positive_weight,
    )
    trainer = Trainer(model, config.training, device, run_dir, positive_weight)
    if resume:
        trainer.resume(resume)
    trainer.fit(loaders["train"], loaders["validation"])

    targets, probabilities = predict(model, loaders["test"], device, "Testing best checkpoint")
    test_metrics = binary_metrics(targets, probabilities, trainer.best_threshold)
    baseline_metrics = {}
    test_frame = subsets["test"]
    for model_name, column in (("gsdti", "gsdti_probability"), ("dcgat", "dcgat_probability")):
        baseline_metrics[model_name] = binary_metrics(
            test_frame["target"].to_numpy(), test_frame[column].to_numpy(), 0.5
        )
    report = {
        "ensemble": test_metrics,
        "baselines": baseline_metrics,
        "test_examples": len(test_frame),
    }
    write_metrics(run_dir / "test_metrics.json", report)
    predictions = test_frame.copy()
    predictions["ensemble_probability"] = probabilities
    predictions["ensemble_label"] = (probabilities >= trainer.best_threshold).astype(np.int8)
    predictions.to_parquet(run_dir / "test_predictions.parquet", index=False)
    LOGGER.info("Test metrics:\n%s", json.dumps(report, indent=2))
    LOGGER.info("Training complete. Best checkpoint: %s", run_dir / "best.pt")
    return run_dir


def _inference_frame(raw: pd.DataFrame, config: EnsembleConfig) -> tuple[pd.DataFrame, bool]:
    wide_required = {"smiles", "sequence", *BASE_FEATURE_COLUMNS}
    if wide_required.issubset(raw.columns):
        frame = raw.copy()
        had_target = "target" in frame.columns or config.data.target_column in frame.columns
        if "target" not in frame.columns:
            if config.data.target_column in frame.columns:
                frame = frame.rename(columns={config.data.target_column: "target"})
            else:
                frame["target"] = 0.0
        return frame[["smiles", "sequence", "target"] + BASE_FEATURE_COLUMNS], had_target
    had_target = config.data.target_column in raw.columns
    if not had_target:
        raw = raw.copy()
        raw[config.data.target_column] = 0.0
    return build_paired_frame(raw, config.data), had_target


def predict_pipeline(
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    device_override: str | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    run_dir = checkpoint_path.parent
    config = load_config(run_dir / "resolved_config.yaml")
    configure_logging(run_dir, config.runtime.log_level)
    if device_override:
        config.runtime.device = device_override
    device = resolve_device(config.runtime.device)
    frame, had_target = _inference_frame(read_table(input_path), config)
    smiles_embeddings = load_or_compute_embeddings(
        frame["smiles"].tolist(), "smiles", config.smiles_encoder, config.runtime.cache_dir, device
    )
    sequence_embeddings = load_or_compute_embeddings(
        frame["sequence"].tolist(), "sequence", config.protein_encoder, config.runtime.cache_dir, device
    )
    dataset = EnsembleDataset(frame, smiles_embeddings, sequence_embeddings)
    loader = make_loader(dataset, config.training, shuffle=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state"]
    model = EnsembleModel(
        smiles_embeddings.values.shape[1],
        sequence_embeddings.values.shape[1],
        config.model,
        state["base_mean"].cpu(),
        state["base_std"].cpu(),
    ).to(device)
    model.load_state_dict(state)
    _, probabilities = predict(model, loader, device, "Ensemble inference")
    threshold = float(checkpoint.get("threshold", 0.5))
    result = frame.copy()
    if not had_target:
        result = result.drop(columns="target")
    result["ensemble_probability"] = probabilities
    result["ensemble_label"] = (probabilities >= threshold).astype(np.int8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".parquet", ".pq"}:
        result.to_parquet(output_path, index=False)
    else:
        result.to_csv(output_path, index=False)
    LOGGER.info("Wrote %d predictions to %s", len(result), output_path)
    return output_path
