import pandas as pd
import pytest

from ensemble.config import DataConfig, SplitConfig
from ensemble.data import BASE_FEATURE_COLUMNS, build_paired_frame, create_splits


def _row(smiles, sequence, target, source, label, probability):
    return {
        "smiles": smiles,
        "sequence": sequence,
        "real_value": target,
        "source": source,
        "label": label,
        "probability": probability,
    }


def test_inner_pairs_repeated_rows_by_target_and_occurrence():
    raw = pd.DataFrame(
        [
            _row("CC", "AAA", 1, "GSDTI", 1, 0.9),
            _row("CC", "AAA", 1, "GSDTI", 0, 0.4),
            _row("CC", "AAA", 0, "GSDTI", 0, 0.1),
            _row("NN", "BBB", 1, "GSDTI", 1, 0.8),
            _row("CC", "AAA", 1, "DCGAT-DTI", 1, 0.7),
            _row("CC", "AAA", 0, "DCGAT-DTI", 0, 0.2),
        ]
    )
    paired = build_paired_frame(raw, DataConfig())
    assert len(paired) == 2
    assert paired["target"].tolist() == [1, 0]
    assert paired.loc[0, "gsdti_probability"] == pytest.approx(0.9)
    assert paired.loc[0, "dcgat_probability"] == pytest.approx(0.7)
    assert list(paired.columns) == ["smiles", "sequence", "target"] + BASE_FEATURE_COLUMNS


@pytest.mark.parametrize("strategy", ["stratified_random", "drug_cold", "target_cold"])
def test_split_assigns_every_row_and_keeps_both_classes(strategy):
    frame = pd.DataFrame(
        {
            "smiles": [f"D{i // 2}" for i in range(120)],
            "sequence": [f"P{i // 3}" for i in range(120)],
            "target": [i % 2 for i in range(120)],
        }
    )
    result = create_splits(
        frame,
        SplitConfig(strategy=strategy, validation_fraction=0.2, test_fraction=0.2, seed=9),
    )
    assert set(result["split"]) == {"train", "validation", "test"}
    assert result.groupby("split")["target"].nunique().eq(2).all()
    if strategy == "drug_cold":
        groups = [set(group["smiles"]) for _, group in result.groupby("split")]
        assert not groups[0] & groups[1]
        assert not groups[0] & groups[2]
        assert not groups[1] & groups[2]
    if strategy == "target_cold":
        groups = [set(group["sequence"]) for _, group in result.groupby("split")]
        assert not groups[0] & groups[1]
        assert not groups[0] & groups[2]
        assert not groups[1] & groups[2]


def test_rejects_out_of_range_probability():
    raw = pd.DataFrame(
        [
            _row("CC", "AAA", 1, "GSDTI", 1, 1.2),
            _row("CC", "AAA", 1, "DCGAT-DTI", 1, 0.8),
        ]
    )
    with pytest.raises(ValueError, match="probabilities"):
        build_paired_frame(raw, DataConfig())
