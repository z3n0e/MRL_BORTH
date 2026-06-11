from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch import nn


class SmallMNISTCNN(nn.Module):
    """Small CNN encoder with an explicit penultimate embedding layer."""

    def __init__(self, feature_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, feature_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        z = self.fc(x)
        return z


class SingleScaleClassifier(nn.Module):
    """Standard classifier: encoder + one linear head on full embedding."""

    def __init__(self, feature_dim: int = 16, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        self.encoder = SmallMNISTCNN(feature_dim=feature_dim, dropout=dropout)
        # Bias is disabled to make NC3 easier to interpret.
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
        self.prefix_dims = [feature_dim]

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        z = self.encoder(x)
        return self.logits_from_features(z)

    def logits_from_features(self, z: torch.Tensor) -> Dict[int, torch.Tensor]:
        return {self.encoder.feature_dim: self.classifier(z)}

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def classifier_weight(self, dim: int | None = None) -> torch.Tensor:
        return self.classifier.weight


class MRLClassifier(nn.Module):
    """Minimal MRL classifier with independent heads for each prefix dimension."""

    def __init__(
        self,
        feature_dim: int = 64,
        prefix_dims: Iterable[int] = (2, 4, 8, 16, 32, 64),
        num_classes: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        prefix_dims = sorted(set(int(d) for d in prefix_dims))
        if len(prefix_dims) == 0:
            raise ValueError("prefix_dims cannot be empty")
        if prefix_dims[-1] != feature_dim:
            raise ValueError("The largest prefix dimension must equal feature_dim")
        if prefix_dims[0] <= 0:
            raise ValueError("Prefix dimensions must be positive")

        self.encoder = SmallMNISTCNN(feature_dim=feature_dim, dropout=dropout)
        self.prefix_dims: List[int] = prefix_dims
        self.heads = nn.ModuleDict(
            {str(d): nn.Linear(d, num_classes, bias=False) for d in prefix_dims}
        )

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        z = self.encoder(x)
        return self.logits_from_features(z)

    def logits_from_features(self, z: torch.Tensor) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        for d in self.prefix_dims:
            out[d] = self.heads[str(d)](z[:, :d])
        return out

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def classifier_weight(self, dim: int) -> torch.Tensor:
        return self.heads[str(dim)].weight


def make_model(
    mode: str,
    feature_dim: int,
    num_classes: int,
    prefix_dims: List[int] | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    if mode == "single":
        return SingleScaleClassifier(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
    if mode == "mrl":
        if prefix_dims is None:
            prefix_dims = [feature_dim]
        return MRLClassifier(
            feature_dim=feature_dim,
            prefix_dims=prefix_dims,
            num_classes=num_classes,
            dropout=dropout,
        )
    raise ValueError(f"Unknown mode: {mode}")
