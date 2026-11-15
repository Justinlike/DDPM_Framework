import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import Attention
from timm.models.layers import to_2tuple

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device) * -emb)
        emb = x[:, None] * emb[None, :]

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

class DiTBlock(nn.Module):
    """
    DiT Block with AdaLN zero conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, skip=False, **kwargs):
        super().__init__()
        self.skip_linear = nn.Linear(2*hidden_size, hidden_size) if skip else None
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True, **kwargs)


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
