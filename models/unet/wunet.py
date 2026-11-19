from abc import abstractmethod
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tools.DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D
from models.unet.nn import conv_nd, normalization, zero_module, scale_module, checkpoint, avg_pool_nd, count_flops_attn, timestep_embedding
from models.unet.unet import TimestepEmbeddingSequential, ResBlock, AttentionBlock, QKVAttentionLegacy, QKVAttention, DownSample, UpSample, WaveletDownsample
import numpy as np

class WavUNetModel(nn.Module):
    def __init__(self, 
                 image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, 
                 dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, num_classes=None,
                 use_checkpoint=False, use_fp16=False, num_heads=1, num_head_channels=-1, num_heads_upsample=-1,
                 use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, num_groups=32,
                 bottleneck_attention=True, resample_2d=True, additive_skips=False, decoder_device_thresh=0,
                 use_freq=True, use_conv = False,progressive_input='residual'):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        # self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        # self.num_heads = num_heads
        # self.num_head_channels = num_head_channels
        # self.num_heads_upsample = num_heads_upsample
        self.num_groups = num_groups
        self.bottleneck_attention = bottleneck_attention
        self.devices = None
        self.decoder_device_thresh = decoder_device_thresh
        self.additive_skips = additive_skips
        self.use_freq = use_freq
        self.use_conv = use_conv
        self.progressive_input = progressive_input

        #############################
        # TIMESTEP EMBEDDING layers #
        #############################
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        ###############
        # INPUT block #
        ###############
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbeddingSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        input_pyramid_channels =in_channels
        ds = 1

        ######################################
        # DOWNWARD path - Feature extraction #
        ######################################
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        channels = ch,
                        emb_channels = time_embed_dim,
                        dropout = dropout,
                        out_channels = mult * model_channels,
                        dims = dims,
                        use_checkpoint = use_checkpoint,
                        use_scale_shift_norm = use_scale_shift_norm,
                        num_groups = num_groups,
                        resample_2d = resample_2d,
                        use_freq = self.use_freq,
                        use_conv = self.use_conv,
                    )
                ]
                ch = mult * model_channels

                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint = use_checkpoint,
                            num_heads = num_heads,
                            num_head_channels = num_head_channels,
                            use_new_attention_order = use_new_attention_order,
                            num_groups = num_groups,
                        )
                    )

                self.input_blocks.append(TimestepEmbeddingSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            # adding downsampling operation in every level
            out_ch = ch
            layers = []
            layers.append(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    out_channels = out_ch,
                    dims = dims,
                    use_checkpoint = use_checkpoint,
                    use_scale_shift_norm = use_scale_shift_norm,
                    down = True,
                    num_groups = num_groups,
                    resample_2d = resample_2d,
                    use_freq = self.use_freq,
                    use_conv = self.use_conv,
                )
                if resblock_updown # default False
                else DownSample(
                    ch,
                    dims=dims,
                    out_channels=out_ch,
                    resample_2d=resample_2d,
                    use_conv=self.use_conv,
                    use_freq=self.use_freq
                )
            )
            self.input_blocks.append(TimestepEmbeddingSequential(*layers))

            # 两个下采样分开添加，为了处理残差分支输入时检查tuple
            # 下采样分支, 添加金字塔残差
            layers = []
            if self.progressive_input == "residual":
                layers.append(
                    WaveletDownsample(
                        in_ch = input_pyramid_channels,
                        out_ch = out_ch,
                    )    
                )
                input_pyramid_channels = out_ch
            self.input_blocks.append(TimestepEmbeddingSequential(*layers))

            ch = out_ch
            input_block_chans.append(ch)
            ds *= 2
            self._feature_size += ch

        self.input_block_chans_bk = input_block_chans[:]

        #########################
        # LATENT/ MIDDLE blocks #
        #########################
        self.middle_block = TimestepEmbeddingSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims = dims,
                use_checkpoint = use_checkpoint,
                use_scale_shift_norm = use_scale_shift_norm,
                # down = True,
                num_groups = num_groups,
                resample_2d = resample_2d,
                use_freq = self.use_freq,
                use_conv = self.use_conv,
            ),
            *([AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                num_groups=self.num_groups,
            )] if self.bottleneck_attention else []),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups,
                resample_2d=resample_2d,
                use_freq=self.use_freq,
                use_conv=self.use_conv,
            ),
        )
        self._feature_size += ch

        # skip_channels = input_block_chans.copy()
        # #################################
        # # UPWARD path - feature mapping #
        # #################################
        # self.output_blocks = nn.ModuleList([])
        # # [0:1, 1:2, 2:4] -> [2:4, 1:2, 0:1]
        # for level, mult in list(enumerate(channel_mult))[::-1]:
        #     for i in range(num_res_blocks+1):                                          # Adding Residual blocks
        #         # 前两层resblock，最后一层是upsample

        #         layers = []
        #         # 与 forward 中 hs.pop() 一一对应
        #         skip_ch = skip_channels.pop()
        #         mid_ch = model_channels * mult

        #         # 小波上采样前，对齐对应编码层的高频子带通道
        #         if self.use_freq and level > 0 and i == num_res_blocks:
        #             mid_ch = model_channels * channel_mult[level - 1]

        #         if self.additive_skips:
        #             assert ch == skip_ch, f"additive_skips need the same channels: ch={ch}, skip_ch={skip_ch}"
        #             block_in_ch = ch
        #         else:
        #             # cat([h, new_hs], dim=1) 后的实际通道数
        #             block_in_ch = ch + skip_ch

        #         layers = [
        #             ResBlock(
        #                 block_in_ch,
        #                 time_embed_dim,
        #                 dropout,
        #                 out_channels=mid_ch,
        #                 dims=dims,
        #                 use_checkpoint=use_checkpoint,
        #                 use_scale_shift_norm=use_scale_shift_norm,
        #                 num_groups=self.num_groups,
        #                 resample_2d=resample_2d,
        #                 use_freq=self.use_freq,
        #                 use_conv=self.use_conv,
        #             )
        #         ]
        #         # Adding Attention layers
        #         if ds in attention_resolutions:
        #             layers.append(
        #                 AttentionBlock(
        #                     mid_ch,
        #                     use_checkpoint=use_checkpoint,
        #                     num_heads=num_heads_upsample,
        #                     num_head_channels=num_head_channels,
        #                     use_new_attention_order=use_new_attention_order,
        #                     num_groups=self.num_groups,
        #                 )
        #             )
        #         ch = mid_ch
        #         # Adding upsampling operation
        #         # 每层都加
        #         if i == num_res_blocks:
        #         # # 最后一层不加
        #         # if level and i == num_res_blocks:
        #             out_ch = ch
        #             layers.append(
        #                 ResBlock(
        #                     mid_ch,
        #                     time_embed_dim,
        #                     dropout,
        #                     out_channels=out_ch,
        #                     dims=dims,
        #                     use_checkpoint=use_checkpoint,
        #                     use_scale_shift_norm=use_scale_shift_norm,
        #                     up=True,
        #                     num_groups=self.num_groups,
        #                     resample_2d=resample_2d,
        #                     use_freq=self.use_freq,
        #                     use_conv=self.use_conv,
        #                 )
        #                 if resblock_updown
        #                 else UpSample(
        #                     mid_ch,
        #                     dims=dims,
        #                     out_channels=out_ch,
        #                     resample_2d=resample_2d,
        #                     use_freq=self.use_freq,
        #                     use_conv=self.use_conv,
        #                 )
        #             )
        #             ds //= 2
        #         self.output_blocks.append(TimestepEmbeddingSequential(*layers))
        #         self._feature_size += ch
        #         mid_ch = ch
        
        # assert not skip_channels, f"skip_channels should be empty: {skip_channels}"

        #################################
        # UPWARD path - feature mapping #
        #################################
        self.output_blocks = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = []
                mid_ch = model_channels * mult

                layers.append(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mid_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                        use_freq=self.use_freq,
                        use_conv=self.use_conv,
                    )
                )
                ch = mid_ch

                # 每个 level 的最后一个块负责上采样。
                if i == num_res_blocks:
                    layers.append(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        up=True,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                        use_freq=self.use_freq,
                        use_conv=self.use_conv,
                    )
                    if resblock_updown
                    else UpSample(
                        ch,
                        dims=dims,
                        out_channels=ch,
                        resample_2d=resample_2d,
                        use_freq=self.use_freq,
                        use_conv=self.use_conv,
                    )
                )
                    ds //= 2

                self.output_blocks.append(
                    TimestepEmbeddingSequential(*layers)
                )
                self._feature_size += ch

        ################
        # Out ResBlock #
        ################
        self.out_res = nn.ModuleList([])
        for i in range(num_res_blocks):
            layers = [
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    out_channels=ch,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    num_groups=self.num_groups,
                    resample_2d=resample_2d,
                    use_freq=self.use_freq,
                    use_conv=self.use_conv,
                )
            ]
            self.out_res.append(TimestepEmbeddingSequential(*layers))

        ################
        # OUTPUT block #
        ################
        self.out = nn.Sequential(
            normalization(ch, self.num_groups),
            nn.SiLU(),
            conv_nd(dims, model_channels, out_channels, 3, padding=1),
        )

    def to(self, *args, **kwargs):
        """
        we overwrite the to() method for the case where we
        distribute parts of our model to different devices
        """
        if isinstance(args[0], (list, tuple)) and len(args[0]) > 1:
            assert not kwargs and len(args) == 1
            # distribute to multiple devices
            self.devices = args[0]
            # move first half to first device, second half to second device
            self.input_blocks.to(self.devices[0])
            self.time_embed.to(self.devices[0])
            self.middle_block.to(self.devices[0])  # maybe devices 0
            for k, b in enumerate(self.output_blocks):
                if k < self.decoder_device_thresh:
                    b.to(self.devices[0])
                else:  # after threshold
                    b.to(self.devices[1])
            self.out.to(self.devices[0])
            print(f"distributed UNet components to devices {self.devices}")

        else:  # default behaviour
            super().to(*args, **kwargs)
            if self.devices is None:  # if self.devices has not been set yet, read it from params
                p = next(self.parameters())
                self.devices = [p.device, p.device]

        return self

    # def forward(self, x, timesteps):
    #     """
    #     Apply the model to an input batch.

    #     :param x: an [N x C x H x W x D] Tensor of inputs.
    #     :param timesteps: a 1-D batch of timesteps.
    #     :param zemb: an [N] Tensor of labels, if class-conditional.
    #     :return: an [N x C x ...] Tensor of outputs.
    #     """
    #     hs = [] # 记录空间下采样
    #     wavelet_hs = [] # 记录频域下采样

    #     input_pyramid = x
    #     emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
    #     h = x
    #     self.hs_shape = []

    #     for module in self.input_blocks:
    #         # print("input module:", module)
    #         if not isinstance(module[0], WaveletDownsample):
    #             h = module(h, emb)
    #             skip = None
    #             if isinstance(h, tuple):
    #                 h, skip = h
    #                 wavelet_hs.append(skip) if skip is not None and self.use_freq else None
    #             hs.append(h)
    #             self.hs_shape.append(h.shape)
    #         else:
    #             # 下采样会有连续两个，先做DownSample，在做WaveletDownsample
    #             # 下采样主干 DownSample: h = module(h, emb)
    #             # 金字塔分支 WaveletDownsample: input_pyramid = module(input_pyramid, emb)
    #             # h = input_pyramid + h
    #             # print("input_pyramid:", input_pyramid.shape, emb.shape)
    #             input_pyramid = module(input_pyramid, emb)
    #             # print(input_pyramid.shape, h.shape)
    #             input_pyramid = input_pyramid + h
    #             h = input_pyramid

    #     for module in self.middle_block:
    #         # h = module(h, emb)
    #         # h = module(h)
    #         # print("middle module:", module)
    #         if isinstance(module, ResBlock):
    #             # resblock
    #             h = module(h, emb)
    #         else:
    #             # attention block
    #             h = module(h)
    #         if isinstance(h, tuple):
    #             h, skip = h
    #             # 中间层的skip有什么用？

    #     # for _ in hs:
    #     #     if _ is not None:
    #     #         print(_.shape)
    #     #     else:
    #     #         print(_)
    #     for _ in wavelet_hs:
    #         if isinstance(_, tuple):
    #             print("wavelet_hs:", len(_))
    #         else:
    #             print("wavelet_hs:", _.shape)

    #     for module in self.output_blocks:
    #         # print("output module:", module)
    #         new_hs = hs.pop()
    #         # new_wavelet_hs = wavelet_hs.pop() if wavelet_hs else None

    #         # if new_hs is not None:
    #         #     skip = new_hs

    #         if h is not None and isinstance(h, tuple):
    #             h = h[0]

    #         assert h.shape[2:] == new_hs.shape[2:], (
    #             f"skip 空间尺寸不匹配: h={h.shape}, skip={new_hs.shape}"
    #         )
    #         # use additive skip connection
    #         # 处理空间域skip
    #         if self.additive_skips:
    #             h = (h + new_hs) / np.sqrt(2.0)
    #         # use frequency aware skip connection
    #         # 这个地方貌似不是我们理解的use_freq，而是指是否使用频域的skip connection
    #         # elif self.use_freq:
    #         elif False:
    #             if isinstance(h, tuple):
    #                 l = list(h)
    #                 l[1] = skip
    #                 h = tuple(l)
    #             else:
    #                 print("进入tuple")
    #                 print("h shape:", h.shape, "new_hs shape:", len(new_wavelet_hs))
    #                 h = (h, new_wavelet_hs)
    #         else:
    #             # print("h shape:", h.shape, "new_hs shape:", new_hs.shape)
    #             h = torch.cat([h, new_hs], dim=1)

    #         # 只有包含小波上采样的块，才取出并携带高频子带
    #         needs_details = any(
    #             isinstance(layer, UpSample) and layer.use_freq
    #             for layer in module
    #         )

    #         if needs_details:
    #             h = (h, wavelet_hs.pop())

    #         h = module(h, emb)

    #     for module in self.out_res:
    #         h = module(h, emb)

    #     h, _ = h
    #     return self.out(h)

    def forward(self, x, timesteps):
        wavelet_hs = []
        input_pyramid = x

        emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        )
        h = x
        self.hs_shape = []

        ################
        # ENCODER path #
        ################
        for module in self.input_blocks:
            if isinstance(module[0], WaveletDownsample):
                input_pyramid = module(input_pyramid, emb)
                input_pyramid = input_pyramid + h
                h = input_pyramid
            else:
                h = module(h, emb)

                if isinstance(h, tuple):
                    h, skip = h
                    if skip is not None:
                        wavelet_hs.append(skip)

                self.hs_shape.append(h.shape)

        ################
        # MIDDLE block #
        ################
        for module in self.middle_block:
            if isinstance(module, ResBlock):
                h = module(h, emb)
            else:
                h = module(h)

            if isinstance(h, tuple):
                h, _ = h

        ################
        # DECODER path #
        ################
        for module in self.output_blocks:
            if isinstance(h, tuple):
                h = h[0]

            needs_details = any(
                (
                    isinstance(layer, UpSample)
                    and layer.use_freq
                )
                or (
                    isinstance(layer, ResBlock)
                    and layer.use_freq
                    and layer.up
                )
                for layer in module
            )


            if needs_details:
                h = (h, wavelet_hs.pop())

            h = module(h, emb)

        ################
        # Out ResBlock #
        ################
        for module in self.out_res:
            h = module(h, emb)

        if isinstance(h, tuple):
            h = h[0]

        return self.out(h)

if __name__ == "__main__":
    from torchprofile import profile_macs

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = torch.randn(1, 1, 32, 256, 256).to(device)
    timesteps = torch.tensor([1], device=device)
    # x = torch.randn(1, 1, 8, 64, 64, device=device)
    print(f"input shape {x.shape}")

    model = WavUNetModel(
        image_size=256,
        in_channels=1,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=[],
        dropout=0.1,
        channel_mult=(1, 2, 2, 4),
        conv_resample=True,
        dims=3,
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
        use_freq = True,
        use_conv = False
    ).to(device)
    
    # print(model)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    with torch.no_grad():
        macs = profile_macs(model, args=(x, timesteps))
    
    # 采用一次乘法 + 一次加法 = 2 FLOPs 的统计口径
    estimated_flops = 2 * macs

    with torch.no_grad():
        out = model(x, timesteps=timesteps)
    
    print("Output shape:", out.shape)

    print(f"Total parameters:     {total_params / 1e6:.3f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.3f} M")
    print(f"MACs:                 {macs / 1e9:.3f} GMACs")
    print(f"Estimated FLOPs:      {estimated_flops / 1e9:.3f} GFLOPs")
