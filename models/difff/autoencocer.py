import gc
import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution
from monai.networks.blocks.spatialattention import SpatialAttentionBlock
from monai.networks.nets.autoencoderkl import AEKLResBlock, AutoencoderKL
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import MaisiEncoder
from monai.utils import ensure_tuple_rep


class MyMaisiEncoder(nn.Module):
    """
    AutoencoderKL with custom MaisiEncoder and MaisiDecoder.

    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_res_blocks: Number of residual blocks per level.
        num_channels: Sequence of block output channels.
        attention_levels: Indicate which level from num_channels contain an attention block.
        latent_channels: Number of channels in the latent space.
        norm_num_groups: Number of groups for the group norm layers.
        norm_eps: Epsilon for the normalization.
        with_encoder_nonlocal_attn: If True, use non-local attention block in the encoder.
        with_decoder_nonlocal_attn: If True, use non-local attention block in the decoder.
        include_fc: whether to include the final linear layer. Default to False.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: If True, use flash attention for a memory efficient attention mechanism.
        use_checkpointing: If True, use activation checkpointing.
        use_convtranspose: If True, use ConvTranspose to upsample feature maps in decoder.
        num_splits: Number of splits for the input tensor.
        dim_split: Dimension of splitting for the input tensor.
        norm_float16: If True, convert output of MaisiGroupNorm3D to float16 format, default to `False`.
        print_info: Whether to print information, default to `False`.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        num_channels: Sequence[int],
        attention_levels: Sequence[bool],
        latent_channels: int = 3,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = False,
        with_decoder_nonlocal_attn: bool = False,
        include_fc: bool = False,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        use_checkpointing: bool = False,
        use_convtranspose: bool = False,
        num_splits: int = 16,
        dim_split: int = 0,
        norm_float16: bool = False,
        print_info: bool = False,
        save_mem: bool = True,
    ) -> None:
        super().__init__()
        self.encoder: nn.Module = MaisiEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            num_splits=num_splits,
            dim_split=dim_split,
            norm_float16=norm_float16,
            print_info=print_info,
            save_mem=save_mem,
        )

        
        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError("AutoencoderKL expects all num_channels being multiple of norm_num_groups")

        if len(num_channels) != len(attention_levels):
            raise ValueError("AutoencoderKL expects num_channels being same size of attention_levels")

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        self.quant_conv_mu = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.quant_conv_log_sigma = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.latent_channels = latent_channels

    def init_from_pretrained(self, vae: AutoencoderKL) -> None:
        self.encoder.load_state_dict(vae.encoder.state_dict())
        self.quant_conv_mu.load_state_dict(vae.quant_conv_mu.state_dict())
        self.quant_conv_log_sigma.load_state_dict(vae.quant_conv_log_sigma.state_dict())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor
        """

        h = self.encoder(x)

        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return z_mu, z_sigma