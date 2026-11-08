import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        return self.proj(x)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,

    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # self.adaln = nn.

    def forward(self, x):
        # x: (batch_size, seq_len, dim)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class DiT(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size,
        num_classes,
        dim,
        num_heads,
        num_layers,

    ):

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers

    def forward(self, x):
