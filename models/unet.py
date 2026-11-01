import torch
import torch.nn as nn

class DownSample(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class UpSample(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ResBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class UNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resoliutions,
            dropout = 0,
            channel_mult = [1, 2, 4, 8],
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            num_groups=32,
            bottleneck_attention=True,
            resample_2d=True,
            additive_skips=False,
            decoder_device_thresh=0,
            ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resoliutions
        


        



