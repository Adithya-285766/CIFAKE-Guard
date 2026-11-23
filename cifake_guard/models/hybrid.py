"""SFHybrid: spatial-frequency CNN-Transformer hybrid (new model).

Motivation (see docs/RESEARCH.md):
* Generators leave traces in the *frequency* domain (up-sampling / decoder artefacts, Durall et al. 2020;
  Frank et al. 2020). At 32x32 these traces live largely in the high-frequency residual, which the
  image content dominates in RGB space.
* CNNs give strong local inductive bias on small data; self-attention lets the model relate distant
  regions and weigh spatial against spectral evidence.

Architecture
  spatial branch : 3 conv stages with Squeeze-Excite channel attention, no stride in the first stage
                   (early down-sampling would destroy pixel-level artefacts) -> 8x8x128 feature map
  frequency branch: fixed high-pass residual (x - blur(x)) -> per-channel log-magnitude FFT (shifted)
                   -> small CNN that keeps spectral *position* (flatten, not global pooling, because
                   location in the spectrum = frequency) -> one 128-d "frequency token"
  fusion          : [CLS, FREQ, 64 spatial tokens] + learned positions -> 2 transformer blocks -> CLS head

Ablation switches: `use_freq=False` removes the frequency token, `use_transformer=False` replaces
attention fusion with global-average-pooling + concatenation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (CIFAKE_MEAN, CIFAKE_STD, Normalize, SqueezeExcite, TransformerBlock, conv_bn,
                     init_transformer_weights)


def high_pass_residual(x: torch.Tensor) -> torch.Tensor:
    """Image minus its 3x3 box-blurred version (suppresses content, keeps fine-grained traces)."""
    blurred = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    return x - blurred


def log_spectrum(x: torch.Tensor) -> torch.Tensor:
    """Per-channel centred log-magnitude spectrum, standardised per image. Computed in float32."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        spec = torch.fft.fftshift(torch.fft.fft2(x.float(), norm="ortho"), dim=(-2, -1))
        mag = torch.log1p(spec.abs())
        mean = mag.mean(dim=(-2, -1), keepdim=True)
        std = mag.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (mag - mean) / std


class SFHybrid(nn.Module):
    def __init__(self, dim: int = 128, depth: int = 2, num_heads: int = 4, dropout: float = 0.1,
                 use_freq: bool = True, use_transformer: bool = True):
        super().__init__()
        self.use_freq, self.use_transformer = use_freq, use_transformer
        self.normalize = Normalize(CIFAKE_MEAN, CIFAKE_STD)
        self.spatial = nn.Sequential(
            conv_bn(3, 32), conv_bn(32, 32), SqueezeExcite(32), nn.MaxPool2d(2),       # 32 -> 16
            conv_bn(32, 64), conv_bn(64, 64), SqueezeExcite(64), nn.MaxPool2d(2),      # 16 -> 8
            conv_bn(64, dim), SqueezeExcite(dim),                                        # 8x8xdim
        )
        if use_freq:
            self.freq = nn.Sequential(
                conv_bn(3, 32), conv_bn(32, 64, stride=2), conv_bn(64, 32, stride=2),   # 32 -> 8
                nn.Flatten(), nn.Dropout(dropout), nn.Linear(32 * 8 * 8, dim), nn.GELU(),
            )
        n_prefix = 1 + int(use_freq)
        if use_transformer:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_prefix + 64, dim))
            self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads, 2.0, dropout) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, 1)
            for m in [*self.blocks, self.norm, self.head]:
                m.apply(init_transformer_weights)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim * (1 + int(use_freq)), 1))
        self.n_prefix_tokens = n_prefix
        self.cam_layer = "spatial.9"      # output of the final 8x8 stage (after its SE attention)

    def frequency_input(self, x: torch.Tensor) -> torch.Tensor:
        return log_spectrum(high_pass_residual(x))

    def forward(self, x):
        feat = self.spatial(self.normalize(x))                      # (B, dim, 8, 8)
        f_tok = self.freq(self.frequency_input(x)) if self.use_freq else None
        if not self.use_transformer:
            pooled = feat.mean(dim=(-2, -1))
            z = torch.cat([pooled, f_tok], 1) if f_tok is not None else pooled
            return self.head(z).squeeze(1)
        tokens = feat.flatten(2).transpose(1, 2)                    # (B, 64, dim)
        prefix = [self.cls_token.expand(x.size(0), -1, -1)]
        if f_tok is not None:
            prefix.append(f_tok.unsqueeze(1))
        z = torch.cat(prefix + [tokens], dim=1) + self.pos_embed
        for blk in self.blocks:
            z = blk(z)
        return self.head(self.norm(z)[:, 0]).squeeze(1)
