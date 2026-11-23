"""LightCNN: the original from-scratch baseline (architecture unchanged, ~0.6 M parameters)."""
from __future__ import annotations

import torch.nn as nn

from .common import CIFAKE_MEAN, CIFAKE_STD, Normalize, conv_bn


class LightCNN(nn.Module):
    cam_layer = "features.7"       # last 8x8 conv stage; the 4x4 stage gives overly coarse maps

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.normalize = Normalize(CIFAKE_MEAN, CIFAKE_STD)
        self.features = nn.Sequential(
            conv_bn(3, 32), conv_bn(32, 32), nn.MaxPool2d(2),        # 32 -> 16
            conv_bn(32, 64), conv_bn(64, 64), nn.MaxPool2d(2),       # 16 -> 8
            conv_bn(64, 128), conv_bn(128, 128), nn.MaxPool2d(2),    # 8 -> 4
            conv_bn(128, 256),                                        # 4x4
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.features(self.normalize(x))).squeeze(1)
