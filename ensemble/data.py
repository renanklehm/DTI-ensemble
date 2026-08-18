from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset

from .config import DataConfig, SplitConfig

LOGGER = logging.getLogger(__name__)
KEY_COLUMNS = ["smiles", "sequence"]
BASE_FEATURE_COLUMNS = [
    "gsdti_label",
    "gsdti_probability",
    "dcgat_label",
    "dcgat_probability",
]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format: {path.suffix}")


def build_paired_frame(raw: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Pair source predictions and discard occurrences absent from either source."""
    required = {
        config.smiles_column,
        config.sequence_column,
        config.target_column,
        config.source_column,
        config.label_column,
        config.probability_column,
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    selected = raw[list(required)].rename(
        columns={
            config.smiles_column: "smiles",
            config.sequence_column: "sequence",
            config.target_column: "target",
            config.source_column: "source",
            config.label_column: "label",
            config.probability_column: "probability",
        }
    )
    selected["smiles"] = selected["smiles"].astype(str)
    selected["sequence"] = selected["sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    if selected[list(KEY_COLUMNS) + ["target", "label", "probability"]].isna().any().any():
        counts = selected.isna().sum()
        raise ValueError(f"Null values are not allowed: {counts[counts > 0].to_dict()}")
    if not selected["target"].isin([0, 1]).all():
        raise ValueError("Targets must be binary 0/1 values")
    if not selected["label"].isin([0, 1]).all():
        raise ValueError("Base-model labels must be binary 0/1 values")
    if not selected["probability"].between(0.0, 1.0).all():
        raise ValueError("Base-model probabilities must be in [0, 1]")

    expected_sources = {config.gsdti_source, config.dcgat_source}
    unexpected = set(selected["source"].unique()) - expected_sources
    if unexpected:
        raise ValueError(f"Unexpected source values: {sorted(unexpected)}")

    match_columns = KEY_COLUMNS + ["target"]
    selected["occurrence"] = selected.groupby(match_columns + ["source"], sort=False).cumcount()
    gsdti = selected[selected["source"] == config.gsdti_source].rename(
        columns={"label": "gsdti_label", "probability": "gsdti_probability"}
    )
    dcgat = selected[selected["source"] == config.dcgat_source].rename(
        columns={"label": "dcgat_label", "probability": "dcgat_probability"}
    )
    join_columns = match_columns + ["occurrence"]
    paired = gsdti[join_columns + ["gsdti_label", "gsdti_probability"]].merge(
        dcgat[join_columns + ["dcgat_label", "dcgat_probability"]],
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )
    dropped_gsdti = len(gsdti) - len(paired)
    dropped_dcgat = len(dcgat) - len(paired)
    if dropped_gsdti or dropped_dcgat:
        LOGGER.info(
            "Inner pairing dropped %d GSDTI and %d DCGAT-DTI unmatched occurrences",
            dropped_gsdti,
            dropped_dcgat,
        )
    paired = paired.drop(columns="occurrence").reset_index(drop=True)
    if paired.empty:
        raise ValueError("No pair has predictions from both base models")
    LOGGER.info(
        "Aligned %d paired examples (%d positive, %d negative)",
        len(paired),
        int(paired["target"].sum()),
        int((paired["target"] == 0).sum()),
    )
    return paired[KEY_COLUMNS + ["target"] + BASE_FEATURE_COLUMNS]


def _random_split_indices(labels: np.ndarray, config: SplitConfig) -> dict[str, np.ndarray]:
    indices = np.arange(len(labels))
    holdout = config.validation_fraction + config.test_fraction
    train, remainder = train_test_split(
        indices, test_size=holdout, random_state=config.seed, stratify=labels
    )
    test_share = config.test_fraction / holdout
    val, test = train_test_split(
        remainder,
        test_size=test_share,
        random_state=config.seed + 1,
        stratify=labels[remainder],
    )
    return {"train": train, "validation": val, "test": test}


def _group_split_indices(frame: pd.DataFrame, group_column: str, config: SplitConfig) -> dict[str, np.ndarray]:
    holdout = config.validation_fraction + config.test_fraction
    outer = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=config.seed)
    train, remainder = next(outer.split(frame, groups=frame[group_column]))
    test_share = config.test_fraction / holdout
    inner = GroupShuffleSplit(n_splits=1, test_size=test_share, random_state=config.seed + 1)
    val_rel, test_rel = next(inner.split(frame.iloc[remainder], groups=frame.iloc[remainder][group_column]))
    return {"train": train, "validation": remainder[val_rel], "test": remainder[test_rel]}


def create_splits(frame: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    if config.strategy == "stratified_random":
        indices = _random_split_indices(frame["target"].to_numpy(), config)
    else:
        group_column = "smiles" if config.strategy == "drug_cold" else "sequence"
        indices = _group_split_indices(frame, group_column, config)
    assignments = np.full(len(frame), "", dtype=object)
    for name, values in indices.items():
        assignments[values] = name
    if (assignments == "").any():
        raise RuntimeError("Split assignment left rows unassigned")
    result = frame.copy()
    result["split"] = assignments
    summary = result.groupby("split")["target"].agg(["count", "mean"])
    LOGGER.info("Split summary:\n%s", summary.to_string())
    for split_name, group in result.groupby("split"):
        if group["target"].nunique() < 2:
            raise ValueError(f"{split_name} split contains only one target class")
    return result


@dataclass(frozen=True)
class IndexedEmbeddings:
    values: torch.Tensor
    index: dict[str, int]


class EnsembleDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        smiles_embeddings: IndexedEmbeddings,
        sequence_embeddings: IndexedEmbeddings,
    ) -> None:
        self.smiles_indices = torch.tensor(
            [smiles_embeddings.index[value] for value in frame["smiles"]], dtype=torch.long
        )
        self.sequence_indices = torch.tensor(
            [sequence_embeddings.index[value] for value in frame["sequence"]], dtype=torch.long
        )
        self.smiles_embeddings = smiles_embeddings.values
        self.sequence_embeddings = sequence_embeddings.values
        self.base_features = torch.from_numpy(
            frame[BASE_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        )
        self.targets = torch.from_numpy(
            frame["target"].to_numpy(dtype=np.float32, copy=True)
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.smiles_embeddings[self.smiles_indices[index]].float(),
            self.sequence_embeddings[self.sequence_indices[index]].float(),
            self.base_features[index],
            self.targets[index],
        )
