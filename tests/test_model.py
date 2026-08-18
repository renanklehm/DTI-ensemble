import torch

from ensemble.config import ModelConfig
from ensemble.model import EnsembleModel


def test_model_forward_and_backward():
    config = ModelConfig(
        projection_dim=8,
        base_projection_dim=4,
        hidden_dims=[16, 8],
        dropout=0.0,
    )
    model = EnsembleModel(12, 10, config, torch.zeros(4), torch.ones(4))
    logits = model(torch.randn(5, 12), torch.randn(5, 10), torch.rand(5, 4))
    assert logits.shape == (5,)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_base_normalization_is_checkpointed():
    model = EnsembleModel(4, 4, ModelConfig(projection_dim=4, hidden_dims=[4]), torch.arange(4), torch.ones(4))
    state = model.state_dict()
    assert "base_mean" in state
    assert "base_std" in state
