from abc import abstractmethod
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tools.DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D
from models.unet.nn import conv_nd, normalization, zero_module, scale_module, checkpoint, avg_pool_nd, count_flops_attn, timestep_embedding
# 目前的想法是将模型的结构定义为一个抽象类，具体的实现由子类完成。这样可以在不同的任务中复用相同的模型结构，同时也方便进行扩展和修改。
# 其他类只需要做初始化即可，
class AbstractUNet(nn.Module):
    def __init__(
            self, 
            in_channels,
            out_channels,
            final_sigmoid,
            basic_module,
            f_maps=64,
            layer_order="gcr",
            num_groups = 16,
            num_levels = 4,
            conv_kernel_size=3,
            pool_kernel_size=2,
            conv_padding=1,
            conv_upscale=2,
            upsample="default",
            dropout_prob=0.1,
            is3d=True,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        if isinstance(f_maps, int):
            pass
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

    def forward(self, x, return_logits=False):
        output, logits = self._forward_logits(x, )
        if return_logits:
            return output, logits
        
        return output


    
    def _forward_logits(self, x, *args, **kwargs):
        # === encoder ===
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)

            encoder_features.insert(0, x)

        encoder_features = encoder_features[1:]

        # === middle ===
        

        # === decoder ===
        for decoder, encoder_feature in zip(self.decoders, ):
            x = decoder(x, encoder_feature)

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass

class TimestepEmbeddingSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    为了解决在UNet中，时间步嵌入（timestep embedding）需要传递给某些子模块的问题，我们定义了一个新的类TimestepEmbeddingSequential，它继承自nn.Sequential和TimestepBlock。这个类的作用是将时间步嵌入传递给支持它的子模块，从而实现对时间步信息的处理。
    """
    def forward(self, x, emb):
        for layer in self:

            # print("\n========== DEBUG ==========")
            # print("Layer:", layer.__class__.__name__)
            # print("Input shape:", x.shape)
            # print("Input device:", x.device)
            # print("Input dtype:", x.dtype)
            # print("Input contiguous:", x.is_contiguous())
            # print("cuDNN enabled:", torch.backends.cudnn.enabled)

            # check if the layer is TimestepBlock, pass the emb to it.
            # if isinstance(layer, TimestepBlock):
            #     x = layer(x, emb)
            # else:
            #     x = layer(x)
            
            if isinstance(x, tuple) and not isinstance(layer, UpSample):
                features, details = x

                if isinstance(layer, TimestepBlock):
                    features = layer(features, emb)
                else:
                    features = layer(features)

                # 当前普通 ResBlock 可能返回 (out, None)
                if isinstance(features, tuple):
                    features, new_details = features
                    if new_details is not None:
                        details = new_details

                x = (features, details)
            else:
                if isinstance(layer, TimestepBlock):
                    x = layer(x, emb)
                else:
                    x = layer(x)
        return x

class DownSample(nn.Module):
    def __init__(self, channels, use_conv = True, dims = 2, out_channels = None, resample_2d = True, use_freq=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels # default to same number of channels
        self.use_conv = use_conv
        self.dims = dims
        stride = (1, 2, 2) if dims == 3 and resample_2d else 2 # un istropic option.

        self.use_freq = use_freq
        self.dwt = DWT_3D("haar") if use_freq and dims == 3 else None
        
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, stride = stride, padding = 1)
        elif self.use_freq:
            self.conv = self.dwt
        else:
            assert self.channels == self.out_channels
            self.conv = avg_pool_nd(dims, kernel_size = stride, stride = stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_freq:
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.conv(x)
            x = (LLL / 3., (LLH, LHL, LHH, HLL, HLH, HHL, HHH))
        else:
            x = self.conv(x)
        return x


class UpSample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    current input: [B, C, H, W, D], wanted [B, C, D, H, W] watting check and review the code.!!!
    """
    def __init__(self, channels, use_conv = True, dims = 2, out_channels = None, resample_2d = True, use_freq = False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels # default to same number of channels
        self.use_conv = use_conv
        self.dims = dims
        self.resample_2d = resample_2d

        self.use_freq = use_freq
        self.idwt = IDWT_3D("haar") if use_freq and dims == 3 else None

        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        if isinstance(x, tuple):
            x, skip = x[0], x[1]
        assert x.shape[1] == self.channels, f"In UpSampling Module expected input with {self.channels} channels, got {x.shape[1]} channels instead"
        # optional: for 3d image, only upsample the height and width, keep the depth unchanged
        # 考虑到医学图像各向异性，深度或许不需要采样。

        if self.use_freq:
            assert skip is None, f"Expected skip connections for IDWT, got {skip}"
            x = self.idwt(3.*x, skip[0], skip[1], skip[2], skip[3], skip[4], skip[5], skip[6])
        elif self.dims == 3 and self.resample_2d:
            # 只对长、宽维度进行上采样，深度维保持不变
            x = F.interpolate(
                x, (x.shape[2], x.shape[3]*2, x.shape[4]*2), mode = "nearest"
            )
        else:
            # 所有空间维度都扩大2倍
            x = F.interpolate(
                x, scale_factor=2, mode="nearest"
            )
        if self.use_conv:
            x = self.conv(x)
        return x

# 频域感知下采样
class WaveletDownsample(nn.Module):
    def __init__(self, in_ch = None, out_ch = None):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = conv_nd(3, in_ch * 8, out_ch, 3, stride = 1, padding = 1)
        self.dwt = DWT_3D("haar")
    
    def forward(self, x):
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwt(x)
        print("in wavelet downsample:", LLL.shape, LLH.shape, LHL.shape, LHH.shape, HLL.shape, HLH.shape, HHL.shape, HHH.shape)
        print(self.in_ch, self.out_ch)
        x = torch.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim = 1) / 3
        print("in wavelet downsample:", x.shape)
        x = self.conv(x)
        return x

class WaveletUpsample(nn.Module):
    def __init__(self, in_ch = None, out_ch = None):
        super().__init__()
        
class WaveletGatingDownsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class WaveletGatingUpsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ResBlock(TimestepBlock):
    """
    a residual block that can optionaly change the number of channels and the resolution of the input.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels = None,
        use_conv = True,
        use_scale_shift_norm = False,
        dims = 2,
        use_checkpoint = False,
        up = False,
        down = False,
        num_groups = 32,
        resample_2d = True,
        use_freq = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.dims = dims
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_shift_scale_norm = use_scale_shift_norm
        self.num_groups = num_groups
        self.use_freq = use_freq
        self.up = up
        self.down = down

        self.in_layers = nn.Sequential(
            normalization(channels, self.num_groups),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = UpSample(channels, use_conv=use_conv, dims=dims, resample_2d=resample_2d, use_freq=use_freq)
            self.x_upd = UpSample(channels, use_conv=use_conv, dims=dims, resample_2d=resample_2d, use_freq=use_freq)
        elif down:
            self.h_upd = DownSample(channels, use_conv=use_conv, dims=dims, resample_2d=resample_2d, use_freq=use_freq)
            self.x_upd = DownSample(channels, use_conv=use_conv, dims=dims, resample_2d=resample_2d, use_freq=use_freq)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            )
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, self.num_groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),
            ),
        )

        # skip branch
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if isinstance(x, tuple):
            skip = x[1]
        else:
            skip = None

        if self.use_freq:
            if self.updown:
                # 下采样怎么办？
                if self.up:
                    x = x[0]
                h = self.in_layers(x)

                if self.up:
                    h = (h, skip)
                    x = (x, skip)
                h, skip = self.h_upd(h)
                x = self.x_upd(x)
            else:
                if isinstance(x, tuple):
                    x = x[0]
                h = self.in_layers(x)
        elif self.use_conv:
            if self.updown:
                in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
                h = in_rest(x)
                h = self.h_upd(h)
                x = self.x_upd(x)
                h = in_conv(h)
            else:
                h = self.in_layers(x)

        emb_out = self.emb_layers(emb)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_shift_scale_norm:
            # print("use_shift_scale_norm is True.")
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        # add skip connections
        print("in resblock:", h.shape, x.shape)
        print(self.skip_connection)
        out = self.skip_connection(x) + h
        # 只有当skip存在，即使用
        if self.use_conv:
            return out
        # 这里可能遇到接受时是一个元组导致报错。原先就是self.skip_connection(x) + h
        # 但是或许，在Downsample和Upsample已经解决
        elif self.use_freq:
            return out, skip
        else:
            return out


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    """
    def __init__(
        self,
        channels,
        num_heads = 1,
        num_head_channels = -1,
        use_checkpoint = False,
        use_new_attention_order = False,
        num_groups = 32,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels, num_groups)
        self.qkv = conv_nd(1, channels, channels * 3, 1)

        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(
            self._forward, (x, ), self.parameters(), True
        )

    def _forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module which perform QKV attention, Matches legacy QKVAttention + input/ouput heads shaping
    """
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        """
        bs, width, length = qkv.shape
        assert width % 3 == 0, f"width of qkv tensor {width} is not divisible by 3"
        ch = width // (3 * self.num_heads)
        q, k, v = qkv.reshape(bs * self.num_heads, ch * 3, length).chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct, bcs -> bts",
            q * scale, k * scale
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts, bcs -> bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)



class QKVAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv):
        """
        Apply QKV attention to the input tensor.
        qkv [B, 3C, T] where T is the number of tokens (spatial dimensions flattened).

        """
        bs, width, length = qkv.shape
        assert width % 3 == 0, f"width of qkv tensor {width} is not divisible by 3"
        ch = width // (3 * self.num_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct, bcs -> bts",
            (q * scale).view(bs * self.num_heads, ch, length),
            (k * scale).view(bs * self.num_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts, bcs -> bct", weight, v.view(bs * self.num_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNet(nn.Module):
    """
    the full U-Net Model, with attention and timestep embedding.
    Isotropic problem
    """
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            use_conv = True,
            use_freq = False,
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
            # means at decoder's 0 layer will be put in device 2
            decoder_device_thresh=0,
            ):
        super().__init__()

        self.use_conv = use_conv
        self.use_freq = use_freq
        assert not (use_conv and use_freq), "use_conv and use_freq cannot be True at the same time."

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
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        
        self.num_heads = num_heads
        self.num_heads_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.num_groups = num_groups
        self.bottleneck_attention = bottleneck_attention
        self.device = None
        self.decoder_device_thresh = decoder_device_thresh
        self.additive_skips = additive_skips
        self.dims = dims
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList([
            TimestepEmbeddingSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            )
        ])

        self._feature_size = model_channels
        input_block_chans = [model_channels] # 记录model channels，便于后续skip拼接

        ch = model_channels # 记录当前输入的通道数
        ds = 1 # 用来判断到了那一层，每做一次下采样*2, ds表示当前的下采样倍数

        # channel_mult: [1, 2, 2, 4, 8]
        # [64, 128, 128, 256, 512]
        ###############################################################
        # INPUT block
        ###############################################################
        for level, mult in enumerate(channel_mult):
            # num_res_blocks = 2
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

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbeddingSequential(
                        # default: resblock_updown = False
                        ResBlock(
                            channels = ch,
                            emb_channels = time_embed_dim,
                            dropout = dropout,
                            out_channels = out_ch,
                            dims = dims,
                            use_checkpoint = use_checkpoint,
                            use_scale_shift_norm = use_scale_shift_norm,
                            num_groups = num_groups,
                            resample_2d = resample_2d,
                        ) 
                        if resblock_updown else
                        DownSample(ch, conv_resample, dims = dims, resample_2d = resample_2d)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.input_blocks_chans_bk = input_block_chans[:]

        ################################################################
        # Middle block
        ################################################################
        self.middle_block = TimestepEmbeddingSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims = dims,
                use_checkpoint = use_checkpoint,
                use_scale_shift_norm = use_scale_shift_norm,
                num_groups = num_groups,
                resample_2d = resample_2d,
            ),
            *([
                AttentionBlock(
                    ch,
                    use_checkpoint = use_checkpoint,
                    num_heads = num_heads,
                    num_head_channels = num_head_channels,
                    use_new_attention_order = use_new_attention_order,
                    num_groups = num_groups,
                )
            ]) if self.bottleneck_attention else [],
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims = dims,
                use_checkpoint = use_checkpoint,
                use_scale_shift_norm = use_scale_shift_norm,
                num_groups = num_groups,
                resample_2d = resample_2d,
            ),
        )
        self._feature_size += ch

        ####################################################################
        # OUTPUT BLOCKS
        ####################################################################
        # if use unet additive_skip == True; else wunet, additive_skip == False
        self.out_blocks = nn.ModuleList([])
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # if additive_skips, mid_ch is the next of ich.
                # [512, 256], [ich, mid_ch]
                # not additive_skips, mid_ch is the next of ich.
                # [512+512, 512], [ich+mid_ch, mid_ch]
                ich = input_block_chans.pop()
                mid_ch = model_channels * mult if not self.additive_skips else (
                    input_block_chans[-1] if input_block_chans else model_channels
                )
                layers = [
                    ResBlock(
                        ch + ich if not self.additive_skips else ch,
                        time_embed_dim,
                        dropout,
                        out_channels = mid_ch,
                        dims = dims,
                        use_checkpoint = use_checkpoint,
                        use_scale_shift_norm = use_scale_shift_norm,
                        num_groups = num_groups,
                        resample_2d = resample_2d,
                    )
                ]   
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            mid_ch,
                            use_checkpoint = use_checkpoint,
                            num_heads = num_heads_upsample,
                            num_head_channels = num_head_channels,
                            use_new_attention_order = use_new_attention_order,
                            num_groups = num_groups,
                        )
                    )
                ch = mid_ch
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            mid_ch,
                            time_embed_dim,
                            dropout,
                            out_channels = out_ch,
                            dims = dims,
                            use_checkpoint = use_checkpoint,
                            use_scale_shift_norm = use_scale_shift_norm,
                            num_groups = num_groups,
                            resample_2d = resample_2d,
                        )
                        if resblock_updown else
                        UpSample(
                            mid_ch,
                            conv_resample,
                            dims = dims,
                            out_channels = out_ch,
                            resample_2d = resample_2d,
                        )
                    )
                    ds //= 2
                self.out_blocks.append(TimestepEmbeddingSequential(*layers))
                self._feature_size += ch
                mid_ch = ch

        self.out = nn.Sequential(
            normalization(ch, num_groups),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def to(self, *args, **kwargs):
        """
        we overwrite the to() method for the case where we
        distribute parts of our model to different devices
        调用to方法, 将GPU设备写入self.device
        """
        if isinstance(args[0], (list, tuple)) and len(args[0]) > 1:
            assert not kwargs and len(args) == 1
            # distribute to multiple devices
            self.device = args[0]
            # move first half to first device, second half to second device
            self.input_blocks.to(self.device[0])
            self.time_embed.to(self.device[0])
            self.middle_block.to(self.device[0])  # maybe devices 0
            for k, b in enumerate(self.output_blocks):
                if k < self.decoder_device_thresh:
                    b.to(self.device[0])
                else:  # after threshold
                    b.to(self.device[1])
            self.out.to(self.device[0])
            print(f"distributed UNet components to devices {self.device}")
        else:
            # super().to(*args, **kwargs)
            # if self.device is None:
            #     p = next(self.parameters())
            #     self.device = [p.device, p.device]
            super().to(*args, **kwargs)
            self.device = None
        
        return self
    
    def forward(self, x, timesteps=None, y = None):
        """
        Apply the model to an input batch
        """
        before = tuple(x.shape)
        # 因为我们的数据输入和模型要求形状不一致，输出模型前先换轴。
        # 有数据不对齐问题，我们实现的UNet要求输入[B, C, D, H, W]，而现在的输入是[B, C, H, W, D]，所以需要permute
        if self.dims == 3:
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        
        assert (y is not None) == (self.num_classes is not None), "must specify y if and only if the model has num_classes"
        
        hs = [] # 存储每一层的输出，用于skip connection
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert (y.shape) == (x.shape[0],), f"Expected y to have shape ({x.shape[0]},), but got {y.shape}"
            emb = emb + self.label_emb(y)

        h = x
        self.hs_shapes = []
        for module in self.input_blocks:
            h = module(h, emb)
            if isinstance(h, tuple):
                h = h[0]  # 如果h是一个元组，取第一个元素作为输出
            hs.append(h)
            self.hs_shapes.append(h.shape)

        h = self.middle_block(h, emb)

        for k, module in enumerate(self.out_blocks):
            skip_hs = hs.pop()
            ### 
            spatial_mismatch = h.shape[2:] != skip_hs.shape[2:]
            additive_mismatch = self.additive_skips and h.shape != skip_hs.shape

            if spatial_mismatch or additive_mismatch:
                encoder_depths = [shape[2] for shape in self.hs_shapes]
                previous_block = self.out_blocks[k - 1] if k > 0 else self.middle_block

                raise RuntimeError(
                    f"decoder block {k}: h={tuple(h.shape)}, "
                    f"skip={tuple(skip_hs.shape)}, "
                    f"encoder D={encoder_depths}\n"
                    f"previous block:\n{previous_block}"
                )
            ### 
            if self.device is not None and k >= self.decoder_device_thresh:
                decoder_device = self.device[1]
                if k == self.decoder_device_thresh:
                    h = h.to(decoder_device)
                    emb = emb.to(decoder_device)
                skip_hs = skip_hs.to(decoder_device)
            
            if self.additive_skips:
                h = (h + skip_hs) / 2
            else:
                h = torch.cat([h, skip_hs], dim=1)
            h = module(h, emb)
        if self.device is not None:
            h = h.to(self.device[0])

        out = self.out(h)
        if self.dims == 3:
            out = out.permute(0, 1, 3, 4, 2).contiguous()  # (B, C, D, H, W) -> (B, C, H, W, D)
        return out

if __name__ == "__main__":
    from torchprofile import profile_macs

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = torch.randn(1, 1, 8, 512, 512).to(device)
    timesteps = torch.tensor([1], device=device)
    # x = torch.randn(1, 1, 8, 64, 64, device=device)
    print(f"input shape {x.shape}")

    model = UNet(
        image_size = 512,
        in_channels = 1,
        model_channels = 64,
        out_channels = 1,
        num_res_blocks = 2,
        attention_resolutions = [],
        dropout = 0.1,
        channel_mult = [1, 2, 2, 4, 8],
        conv_resample = True,
        dims = 3,
        num_classes = None,
        use_checkpoint = False,
        use_fp16 = False,
        num_heads = 1,
        num_head_channels = -1,
        num_heads_upsample = -1,
        use_scale_shift_norm = False,
        resblock_updown = False,
        use_new_attention_order = False,
        num_groups = 32,
        bottleneck_attention = True,
        additive_skips = True,
    ).to(device)

    # print("================================")
    # print(model)
    # print("================================")

    # model = UNet(
    #     image_size=64,
    #     in_channels=1,
    #     model_channels=32,
    #     out_channels=1,
    #     num_res_blocks=2,
    #     attention_resolutions=[8],
    #     dropout=0.1,
    #     channel_mult=[1, 2, 4, 8],
    #     conv_resample=True,
    #     dims=3,
    #     num_classes=None,
    #     use_checkpoint=False,
    #     use_fp16=False,
    #     num_heads=1,
    #     num_head_channels=-1,
    #     num_heads_upsample=-1,
    #     use_scale_shift_norm=False,
    #     resblock_updown=False,
    #     use_new_attention_order=False,
    #     num_groups=32,
    #     bottleneck_attention=True,
    #     resample_2d=True,
    # ).to(device)

    model.eval()

    with torch.no_grad():
        out = model(x, timesteps=timesteps, y=None)

    print("Output shape:", out.shape)
    # 有个地方不动了
        


        



