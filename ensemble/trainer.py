from __future__ import annotations

import csv
import json
import logging
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import TrainingConfig
from .metrics import binary_metrics, find_optimal_threshold

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False


def make_loader(dataset: Any, config: TrainingConfig, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        generator=generator,
    )


def _cosine_schedule(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, multiplier)


def _move_batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(item.to(device, non_blocking=True) for item in batch)


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    progress = tqdm(loader, desc=description, unit="batch", dynamic_ncols=True, leave=False)
    for batch in progress:
        smiles, sequence, features, targets = _move_batch(batch, device)
        probabilities = torch.sigmoid(model(smiles, sequence, features))
        all_probabilities.append(probabilities.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    return np.concatenate(all_targets), np.concatenate(all_probabilities)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        output_dir: str | Path,
        positive_weight: float,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        pos_weight = torch.tensor([positive_weight], device=device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.use_amp = config.mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.start_epoch = 0
        self.best_score = -math.inf if config.monitor_mode == "max" else math.inf
        self.epochs_without_improvement = 0
        self.best_threshold = 0.5
        self.scheduler: LambdaLR | None = None

    def _is_improved(self, score: float) -> bool:
        if self.config.monitor_mode == "max":
            return score > self.best_score + self.config.min_delta
        return score < self.best_score - self.config.min_delta

    def _save_checkpoint(self, path: Path, epoch: int, metrics: dict[str, float]) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state": self.scaler.state_dict(),
                "best_score": self.best_score,
                "epochs_without_improvement": self.epochs_without_improvement,
                "threshold": self.best_threshold,
                "metrics": metrics,
            },
            path,
        )

    def resume(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_score = float(checkpoint["best_score"])
        self.epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        self.best_threshold = float(checkpoint.get("threshold", 0.5))
        self._resume_scheduler_state = checkpoint.get("scheduler_state")
        LOGGER.info("Resuming at epoch %d from %s", self.start_epoch + 1, checkpoint_path)

    def fit(self, train_loader: DataLoader, validation_loader: DataLoader) -> dict[str, float]:
        updates_per_epoch = math.ceil(
            len(train_loader) / self.config.gradient_accumulation_steps
        )
        total_steps = updates_per_epoch * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_fraction)
        self.scheduler = _cosine_schedule(self.optimizer, warmup_steps, total_steps)
        scheduler_state = getattr(self, "_resume_scheduler_state", None)
        if scheduler_state:
            self.scheduler.load_state_dict(scheduler_state)

        history_path = self.output_dir / "metrics.csv"
        fields = [
            "epoch", "train_loss", "lr", "validation_loss", "validation_auroc",
            "validation_auprc", "validation_accuracy", "validation_balanced_accuracy",
            "validation_f1", "validation_precision", "validation_recall", "validation_mcc",
            "validation_brier", "validation_threshold",
        ]
        write_header = not history_path.exists() or self.start_epoch == 0
        with history_path.open("a" if not write_header else "w", newline="", encoding="utf-8") as history:
            writer = csv.DictWriter(history, fieldnames=fields)
            if write_header:
                writer.writeheader()
            for epoch in range(self.start_epoch, self.config.epochs):
                train_loss = self._train_epoch(train_loader, epoch)
                targets, probabilities = predict(
                    self.model, validation_loader, self.device, "Validating"
                )
                threshold = find_optimal_threshold(
                    targets, probabilities, self.config.threshold_metric
                )
                validation_metrics = binary_metrics(targets, probabilities, threshold)
                row: dict[str, float | int] = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
                writer.writerow(row)
                history.flush()
                LOGGER.info(
                    "Epoch %d/%d | train_loss=%.5f | val_loss=%.5f | AUROC=%.5f | "
                    "AUPRC=%.5f | F1=%.5f | threshold=%.3f",
                    epoch + 1,
                    self.config.epochs,
                    train_loss,
                    validation_metrics["loss"],
                    validation_metrics["auroc"],
                    validation_metrics["auprc"],
                    validation_metrics["f1"],
                    threshold,
                )
                score = validation_metrics[self.config.monitor]
                improved = self._is_improved(score)
                if improved:
                    self.best_score = score
                    self.best_threshold = threshold
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(self.output_dir / "best.pt", epoch, validation_metrics)
                    LOGGER.info("New best %s: %.6f", self.config.monitor, score)
                else:
                    self.epochs_without_improvement += 1
                self._save_checkpoint(self.output_dir / "last.pt", epoch, validation_metrics)
                if self.epochs_without_improvement >= self.config.patience:
                    LOGGER.info("Early stopping after %d unimproved epochs", self.epochs_without_improvement)
                    break
        checkpoint = torch.load(self.output_dir / "best.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.best_threshold = float(checkpoint["threshold"])
        return checkpoint["metrics"]

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        examples = 0
        progress = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        amp_context = (
            lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        ) if self.use_amp else nullcontext
        for step, batch in enumerate(progress):
            smiles, sequence, features, targets = _move_batch(batch, self.device)
            with amp_context():
                logits = self.model(smiles, sequence, features)
                raw_loss = self.criterion(logits, targets)
                loss = raw_loss / self.config.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
            should_update = (
                (step + 1) % self.config.gradient_accumulation_steps == 0
                or step + 1 == len(loader)
            )
            if should_update:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                assert self.scheduler is not None
                self.scheduler.step()
            count = len(targets)
            total_loss += raw_loss.item() * count
            examples += count
            progress.set_postfix(loss=f"{total_loss / examples:.5f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
        return total_loss / examples


def write_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
