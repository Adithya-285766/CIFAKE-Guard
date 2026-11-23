from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAKE_MEAN, CIFAKE_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Normalize(nn.Module):
    """Channel normalisation inside the model, so every model accepts [0, 1] RGB tensors."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class Upsample(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x):
        if x.shape[-1] == self.size and x.shape[-2] == self.size:
            return x
        return F.interpolate(x, size=(self.size, self.size), mode="bicubic", align_corners=False)


def conv_bn(c_in, c_out, stride=1):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class SqueezeExcite(nn.Module):
    """Channel attention (Hu et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class Attention(nn.Module):
    """Multi-head self-attention that can optionally keep its attention matrix for explanation."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop, self.proj_drop = dropout, nn.Dropout(dropout)
        self.keep_attention = False
        self.last_attention: torch.Tensor | None = None

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.keep_attention:
            attn = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
            attn = attn.softmax(dim=-1)
            self.last_attention = attn.detach()
            out = attn @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(b, n, d)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (same structure as nn.TransformerEncoderLayer(norm_first=True))."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def init_transformer_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
