from pathlib import Path

import pandas as pd
import torch

from ensemble.config import EnsembleConfig, ModelConfig, save_config
from ensemble.data import IndexedEmbeddings
from ensemble.model import EnsembleModel
from ensemble.pipeline import predict_pipeline


def test_prediction_pipeline_writes_probabilities(tmp_path: Path, monkeypatch):
    config = EnsembleConfig()
    config.runtime.output_dir = str(tmp_path)
    config.runtime.cache_dir = str(tmp_path / "cache")
    config.runtime.device = "cpu"
    config.training.batch_size = 2
    config.model = ModelConfig(
        projection_dim=4,
        base_projection_dim=3,
        hidden_dims=[8],
        dropout=0.0,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_config(config, run_dir / "resolved_config.yaml")
    frame = pd.DataFrame(
        {
            "smiles": ["CC", "NN"],
            "sequence": ["AAA", "BBB"],
            "gsdti_label": [1, 0],
            "gsdti_probability": [0.8, 0.2],
            "dcgat_label": [1, 0],
            "dcgat_probability": [0.7, 0.3],
        }
    )
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    frame.to_parquet(input_path, index=False)
    drug_values = torch.randn(2, 5)
    protein_values = torch.randn(2, 6)

    def fake_embeddings(values, kind, config, cache_root, device):
        if kind == "smiles":
            return IndexedEmbeddings(drug_values, {"CC": 0, "NN": 1})
        return IndexedEmbeddings(protein_values, {"AAA": 0, "BBB": 1})

    monkeypatch.setattr("ensemble.pipeline.load_or_compute_embeddings", fake_embeddings)
    model = EnsembleModel(5, 6, config.model, torch.zeros(4), torch.ones(4))
    torch.save({"model_state": model.state_dict(), "threshold": 0.4}, run_dir / "best.pt")
    predict_pipeline(run_dir / "best.pt", input_path, output_path)
    result = pd.read_parquet(output_path)
    assert len(result) == 2
    assert "target" not in result.columns
    assert result["ensemble_probability"].between(0, 1).all()
    assert result["ensemble_label"].isin([0, 1]).all()
