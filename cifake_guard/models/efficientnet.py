"""EfficientNet-B0 fine-tuned from ImageNet weights.

Changes w.r.t. the original notebook:
* resizing + ImageNet normalisation moved inside the model (single input contract for all models);
* BUG FIX: while the backbone is frozen, its BatchNorm layers are now kept in eval mode. Previously
  `requires_grad=False` froze the weights but BatchNorm running statistics kept drifting toward CIFAKE
  statistics during the "frozen" epoch.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .common import IMAGENET_MEAN, IMAGENET_STD, Normalize, Upsample


class EfficientNetB0Detector(nn.Module):
    def __init__(self, img_size: int = 128, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        self.resize = Upsample(img_size)
        self.normalize = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.features = net.features
        self.pool = net.avgpool
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(net.classifier[1].in_features, 1))
        self.backbone_frozen = False
        # pick the stage whose feature map is 8x8 (finer than the final stride-32 stage):
        # stride 16 (features.5) for >=128 px inputs, stride 8 (features.3) for 64 px inputs
        self.cam_layer = "features.5" if img_size >= 128 else "features.3"

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.backbone_frozen = not trainable
        for p in self.features.parameters():
            p.requires_grad = trainable
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.features.eval()
        return self

    def param_groups(self, lr: float):
        """Discriminative learning rates: backbone 10x lower than the new head."""
        return [{"params": list(self.features.parameters()), "lr": lr * 0.1},
                {"params": list(self.head.parameters()), "lr": lr}]

    def forward(self, x):
        x = self.normalize(self.resize(x))
        return self.head(self.pool(self.features(x))).squeeze(1)
