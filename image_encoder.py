"""Hierarchical MedViT style image encoder with a feature pyramid network.

The encoder follows the hybrid convolution and transformer design popularised
by MedViT: early stages use Efficient Convolution Blocks (ECB) that capture
local texture cheaply, later stages use Local Transformer Blocks (LTB) whose
spatial reduction attention models long range context at modest cost. A top
down feature pyramid (FPN) merges the three stages so that fine grained detail
(small nodules, nuclei) and global context (lung anatomy, tissue architecture)
are both available to the fusion module and to the localisation head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from medfusion.utils import sub


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


class ConvStem(nn.Module):
    """Two stride two convolutions: overall downsampling by four."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(ConvBNAct(in_ch, out_ch // 2, 3, 2), ConvBNAct(out_ch // 2, out_ch, 3, 2))

    def forward(self, x):
        return self.body(x)


class EfficientConvBlock(nn.Module):
    """ECB: pointwise expansion, depthwise spatial mixing, pointwise projection."""

    def __init__(self, dim: int, expansion: int = 2, drop_path: float = 0.0):
        super().__init__()
        hidden = dim * expansion
        self.block = nn.Sequential(
            ConvBNAct(dim, hidden, 1), ConvBNAct(hidden, hidden, 3, groups=hidden),
            nn.Conv2d(hidden, dim, 1, bias=False), nn.BatchNorm2d(dim),
        )
        self.drop_path = drop_path

    def forward(self, x):
        out = self.block(x)
        if self.training and self.drop_path > 0:
            keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.drop_path
            out = out * keep / sub(1.0, self.drop_path)
        return x + out


class SpatialReductionAttention(nn.Module):
    """Multi head self attention whose keys and values are spatially pooled."""

    def __init__(self, dim: int, heads: int, reduction: int = 1, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** 0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.reduction = reduction
        if reduction > 1:
            self.reduce = nn.Sequential(nn.Conv2d(dim, dim, reduction, reduction), nn.BatchNorm2d(dim))
        self.dropout = nn.Dropout(dropout)
        self.last_attention = None

    def forward(self, x, h: int, w: int):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        if self.reduction > 1:
            pooled = self.reduce(x.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)
        else:
            pooled = x
        kv = self.kv(pooled).reshape(b, pooled.shape[1], 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = torch.softmax((q @ k.transpose(~1, ~0)) / self.scale, dim=~0)
        self.last_attention = attn.detach()
        out = (self.dropout(attn) @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class LocalFeedForward(nn.Module):
    """Convolutional feed forward network that preserves locality."""

    def __init__(self, dim: int, expansion: int = 3):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.act = nn.GELU()

    def forward(self, x, h, w):
        b, n, c = x.shape
        y = x.transpose(1, 2).reshape(b, c, h, w)
        y = self.fc2(self.act(self.dw(self.act(self.fc1(y)))))
        return y.flatten(2).transpose(1, 2)


class LocalTransformerBlock(nn.Module):
    """LTB: spatial reduction self attention followed by a local feed forward."""

    def __init__(self, dim: int, heads: int, reduction: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, heads, reduction, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = LocalFeedForward(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.attn(self.norm1(tokens), h, w)
        tokens = tokens + self.ffn(self.norm2(tokens), h, w)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


def sinusoidal_2d(h: int, w: int, dim: int):
    """Fixed 2D sinusoidal position encoding of shape (h * w, dim)."""
    quarter = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float32) / quarter))
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    out_y = ys.flatten()[:, None] * omega[None]
    out_x = xs.flatten()[:, None] * omega[None]
    return torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1)


class HierarchicalMedViT(nn.Module):
    """Three stage hybrid encoder with a feature pyramid.

    Returns pyramid level tokens for fusion, the finest pyramid map for dense
    prediction, and a pooled global descriptor.
    """

    def __init__(self, in_channels: int, image_size: int, channels=(32, 64, 128), d_model: int = 128,
                 heads: int = 4, dropout: float = 0.0):
        super().__init__()
        c1, c2, c3 = channels
        self.stem = ConvStem(in_channels, c1)
        self.stage1 = nn.Sequential(EfficientConvBlock(c1), EfficientConvBlock(c1))
        self.merge2 = ConvBNAct(c1, c2, 3, 2)
        self.stage2 = nn.Sequential(EfficientConvBlock(c2), LocalTransformerBlock(c2, heads, reduction=2, dropout=dropout))
        self.merge3 = ConvBNAct(c2, c3, 3, 2)
        self.stage3 = nn.Sequential(LocalTransformerBlock(c3, heads, 1, dropout), LocalTransformerBlock(c3, heads, 1, dropout))
        self.lateral = nn.ModuleList([nn.Conv2d(c, d_model, 1) for c in (c1, c2, c3)])
        self.smooth = nn.ModuleList([nn.Sequential(ConvBNAct(d_model, d_model, 3, groups=d_model), ConvBNAct(d_model, d_model, 1))
                                     for _ in range(2)])
        g3 = image_size // 16
        self.grid = g3
        self.register_buffer("pos", sinusoidal_2d(g3, g3, d_model), persistent=False)
        self.level_embed = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.level_embed, std=0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        f1 = self.stage1(self.stem(x))
        f2 = self.stage2(self.merge2(f1))
        f3 = self.stage3(self.merge3(f2))
        p3 = self.lateral[2](f3)
        p2 = self.smooth[0](self.lateral[1](f2) + F.interpolate(p3, size=f2.shape[2:], mode="nearest"))
        p1 = self.smooth[1](self.lateral[0](f1) + F.interpolate(p2, size=f1.shape[2:], mode="nearest"))
        coarse = p3.flatten(2).transpose(1, 2) + self.pos + self.level_embed[0]
        mid = F.avg_pool2d(p2, 2).flatten(2).transpose(1, 2) + self.pos + self.level_embed[1]
        tokens = self.norm(torch.cat([coarse, mid], dim=1))
        return {"tokens": tokens, "fine_map": p1, "mid_map": p2, "pooled": tokens.mean(dim=1)}

    def attention_maps(self):
        """Self attention maps from the transformer stages (for rollout style XAI)."""
        maps = []
        for module in self.modules():
            if isinstance(module, SpatialReductionAttention) and module.last_attention is not None:
                maps.append(module.last_attention)
        return maps
