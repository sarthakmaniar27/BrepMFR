from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ReconstructionLosses:
    continuous: torch.Tensor
    categorical: torch.Tensor


class MaskedGeometryHead(nn.Module):
    """Predict intrinsic face attributes from a contextually encoded masked face."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.continuous = nn.Linear(hidden_dim, 10)
        self.face_type = nn.Linear(hidden_dim, 8)
        self.face_loop = nn.Linear(hidden_dim, 256)
        self.degree = nn.Linear(hidden_dim, 128)

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(embeddings)
        return {
            "continuous": self.continuous(hidden),
            "face_type": self.face_type(hidden),
            "face_loop": self.face_loop(hidden),
            "degree": self.degree(hidden),
        }

    def loss(
        self,
        predictions: dict[str, torch.Tensor],
        *,
        continuous_target: torch.Tensor,
        face_type: torch.Tensor,
        face_loop: torch.Tensor,
        degree: torch.Tensor,
    ) -> ReconstructionLosses:
        continuous = F.smooth_l1_loss(predictions["continuous"], continuous_target)
        categorical = (
            F.cross_entropy(predictions["face_type"], face_type.long().clamp(0, 7))
            + F.cross_entropy(predictions["face_loop"], face_loop.long().clamp(0, 255))
            + F.cross_entropy(predictions["degree"], degree.long().clamp(0, 127))
        ) / 3.0
        return ReconstructionLosses(continuous=continuous, categorical=categorical)

