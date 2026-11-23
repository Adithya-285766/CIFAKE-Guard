"""TinyViT: the original from-scratch Vision Transformer for 32x32 inputs (~1.8 M parameters).

Same architecture as the notebook (4x4 patches -> 64 tokens + [CLS], 6 pre-norm blocks, dim 192,
3 heads, MLP ratio 2), re-implemented with explicit attention modules so attention maps can be
extracted for attention-rollout explanations (nn.TransformerEncoder does not expose them).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import CIFAKE_MEAN, CIFAKE_STD, Normalize, TransformerBlock, init_transformer_weights


class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=3, mlp_ratio=2.0,
                 dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = img_size // patch_size
        self.normalize = Normalize(CIFAKE_MEAN, CIFAKE_STD)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid ** 2 + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)
        self.apply(init_transformer_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.n_prefix_tokens = 1       # tokens before the spatial grid (for attention rollout)
        self.cam_layer = None          # no conv feature map; use attention rollout instead

    def forward(self, x):
        x = self.patch_embed(self.normalize(x)).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0]).squeeze(1)
