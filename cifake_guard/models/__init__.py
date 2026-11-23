"""Model registry. `build_model(spec)` turns a config entry into an nn.Module."""
from __future__ import annotations

import torch.nn as nn

from .cnn import LightCNN
from .efficientnet import EfficientNetB0Detector
from .hybrid import SFHybrid
from .vit import TinyViT

ARCHS = {
    "light_cnn": lambda s: LightCNN(dropout=s.get("dropout", 0.3)),
    "efficientnet_b0": lambda s: EfficientNetB0Detector(img_size=s.get("img_size", 128),
                                                        pretrained=s.get("pretrained", True),
                                                        dropout=s.get("dropout", 0.3)),
    "tiny_vit": lambda s: TinyViT(depth=s.get("depth", 6), embed_dim=s.get("embed_dim", 192),
                                  num_heads=s.get("num_heads", 3), dropout=s.get("dropout", 0.1)),
    "sf_hybrid": lambda s: SFHybrid(dim=s.get("dim", 128), depth=s.get("depth", 2),
                                    use_freq=s.get("use_freq", True),
                                    use_transformer=s.get("use_transformer", True),
                                    dropout=s.get("dropout", 0.1)),
}


def build_model(spec: dict, pretrained_override: bool | None = None) -> nn.Module:
    spec = dict(spec)
    if pretrained_override is not None:
        spec["pretrained"] = pretrained_override
    arch = spec["arch"]
    if arch not in ARCHS:
        raise ValueError(f"Unknown architecture {arch!r}")
    return ARCHS[arch](spec)


__all__ = ["build_model", "ARCHS", "LightCNN", "EfficientNetB0Detector", "TinyViT", "SFHybrid"]
