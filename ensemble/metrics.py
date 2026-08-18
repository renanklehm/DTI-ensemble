from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_optimal_threshold(
    targets: np.ndarray, probabilities: np.ndarray, metric: str = "f1"
) -> float:
    if metric not in {"f1", "balanced_accuracy", "mcc"}:
        raise ValueError("threshold_metric must be f1, balanced_accuracy, or mcc")
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 199), probabilities]))
    if len(candidates) > 2000:
        candidates = np.quantile(candidates, np.linspace(0, 1, 2000))
    scorer = {
        "f1": f1_score,
        "balanced_accuracy": balanced_accuracy_score,
        "mcc": matthews_corrcoef,
    }[metric]
    scores = [scorer(targets, probabilities >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def binary_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    clipped = np.clip(probabilities, 1.0e-7, 1.0 - 1.0e-7)
    return {
        "loss": float(log_loss(targets, clipped, labels=[0, 1])),
        "auroc": float(roc_auc_score(targets, probabilities)),
        "auprc": float(average_precision_score(targets, probabilities)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(targets, predictions)),
        "brier": float(brier_score_loss(targets, probabilities)),
        "threshold": float(threshold),
    }
