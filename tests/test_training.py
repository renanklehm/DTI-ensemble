from pathlib import Path
import warnings

import pandas as pd
import torch

from ensemble.config import ModelConfig, TrainingConfig
from ensemble.data import EnsembleDataset, IndexedEmbeddings
from ensemble.model import EnsembleModel
from ensemble.trainer import Trainer, make_loader, predict, seed_everything


def test_cpu_training_checkpoint_and_resume(tmp_path: Path):
    seed_everything(3)
    rows = 48
    frame = pd.DataFrame(
        {
            "smiles": [f"D{i}" for i in range(rows)],
            "sequence": [f"P{i}" for i in range(rows)],
            "target": [i % 2 for i in range(rows)],
            "gsdti_label": [i % 2 for i in range(rows)],
            "gsdti_probability": [0.8 if i % 2 else 0.2 for i in range(rows)],
            "dcgat_label": [i % 2 for i in range(rows)],
            "dcgat_probability": [0.7 if i % 2 else 0.3 for i in range(rows)],
        }
    )
    drugs = torch.randn(rows, 6)
    proteins = torch.randn(rows, 7)
    dataset = EnsembleDataset(
        frame,
        IndexedEmbeddings(drugs, {f"D{i}": i for i in range(rows)}),
        IndexedEmbeddings(proteins, {f"P{i}": i for i in range(rows)}),
    )
    config = TrainingConfig(
        epochs=2,
        batch_size=12,
        learning_rate=0.01,
        patience=2,
        mixed_precision=False,
        pin_memory=False,
    )
    loader = make_loader(dataset, config, shuffle=True)
    model_config = ModelConfig(projection_dim=8, base_projection_dim=4, hidden_dims=[16], dropout=0.0)
    model = EnsembleModel(6, 7, model_config, dataset.base_features.mean(0), dataset.base_features.std(0))
    trainer = Trainer(model, config, torch.device("cpu"), tmp_path, positive_weight=1.0)
    trainer.fit(loader, make_loader(dataset, config, shuffle=False))
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "metrics.csv").exists()
    targets, probabilities = predict(model, make_loader(dataset, config, False), torch.device("cpu"), "test")
    assert targets.shape == probabilities.shape == (rows,)
    assert ((0 <= probabilities) & (probabilities <= 1)).all()

    resumed_model = EnsembleModel(6, 7, model_config, torch.zeros(4), torch.ones(4))
    resumed = Trainer(resumed_model, config, torch.device("cpu"), tmp_path, positive_weight=1.0)
    resumed.resume(tmp_path / "last.pt")
    assert resumed.start_epoch == 2


def test_dataset_owns_writable_numpy_storage():
    frame = pd.DataFrame(
        {
            "smiles": ["D"],
            "sequence": ["P"],
            "target": [1],
            "gsdti_label": [1],
            "gsdti_probability": [0.8],
            "dcgat_label": [1],
            "dcgat_probability": [0.7],
        }
    )
    embeddings = IndexedEmbeddings(torch.ones(1, 2), {"D": 0})
    proteins = IndexedEmbeddings(torch.ones(1, 2), {"P": 0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dataset = EnsembleDataset(frame, embeddings, proteins)
    assert dataset.base_features.numpy().flags.writeable
    assert dataset.targets.numpy().flags.writeable
