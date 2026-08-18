from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class ProjectionBlock(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )


class EnsembleModel(nn.Module):
    """Fuse molecular/protein representations with two base-model predictions."""

    def __init__(
        self,
        smiles_dim: int,
        sequence_dim: int,
        config: ModelConfig,
        base_mean: torch.Tensor,
        base_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.smiles_dim = smiles_dim
        self.sequence_dim = sequence_dim
        self.config = config
        self.smiles_projection = ProjectionBlock(smiles_dim, config.projection_dim, config.dropout)
        self.sequence_projection = ProjectionBlock(sequence_dim, config.projection_dim, config.dropout)
        self.base_projection = ProjectionBlock(8, config.base_projection_dim, config.dropout)
        self.register_buffer("base_mean", base_mean.float().reshape(1, 4))
        self.register_buffer("base_std", base_std.float().reshape(1, 4).clamp_min(1.0e-6))

        fusion_dim = config.projection_dim * 4 + config.base_projection_dim
        layers: list[nn.Module] = []
        current = fusion_dim
        for hidden in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(current, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            current = hidden
        layers.append(nn.Linear(current, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        smiles_embedding: torch.Tensor,
        sequence_embedding: torch.Tensor,
        base_features: torch.Tensor,
    ) -> torch.Tensor:
        smiles = self.smiles_projection(smiles_embedding)
        sequence = self.sequence_projection(sequence_embedding)
        normalized = (base_features - self.base_mean) / self.base_std
        gsdti_label, gsdti_probability, dcgat_label, dcgat_probability = base_features.unbind(dim=1)
        derived = torch.stack(
            [
                (gsdti_probability - dcgat_probability).abs(),
                (gsdti_probability + dcgat_probability) * 0.5,
                (gsdti_label == dcgat_label).to(base_features.dtype),
                (gsdti_label + dcgat_label) * 0.5,
            ],
            dim=1,
        )
        base = self.base_projection(torch.cat([normalized, derived], dim=1))
        joint = torch.cat(
            [smiles, sequence, (smiles - sequence).abs(), smiles * sequence, base], dim=1
        )
        return self.classifier(joint).squeeze(1)
