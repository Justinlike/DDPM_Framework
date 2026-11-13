import timm
# import segmentation_models_pytorch as smp
from .semi_ct import CTCo, CTCo_diff
# from .Leaf import *
import json
from .jct import JCT
from .jct_noise import JCT_noise
from .jct2d import JCT2D, JCT2D_noise
from .unet import UNet

def build_model(cfg):
    model_name = cfg.model.lower()
    sample_steps = cfg.sample_steps
    sample_method = cfg.sample_method

    if model_name == "jct":
        with open('/data/jzk/CT/inference.json') as f:
            conf = json.load(f)
        model = JCT(conf, num_timesteps=sample_steps, method=sample_method)
    elif model_name == 'jct_noise':
        with open('/data/jzk/CT/inference.json') as f:
            conf = json.load(f)
        model = JCT_noise(conf)
    elif model_name == 'jct2d':
        model = JCT2D(cfg)
    elif model_name == 'jct2d_noise':
        model = JCT2D_noise(cfg)
    elif model_name == "jct_unet":
        # with open("/data/jzk/DDPM_Framework/conf/config-jct_unet.json") as f:
        #     conf = json.load(f)
        # UNet 使用自己实现的版本
        unet = UNet(
        image_size = cfg.model_config.image_size,
        in_channels = cfg.model_config.in_channels,
        model_channels = cfg.model_config.model_channels,
        out_channels = cfg.model_config.out_channels,
        num_res_blocks = cfg.model_config.num_res_blocks,
        attention_resolutions = cfg.model_config.attention_resolutions,
        dropout = cfg.model_config.dropout,
        channel_mult = cfg.model_config.channel_mult,
        conv_resample = cfg.model_config.conv_resample,
        dims = cfg.model_config.dims,
        num_classes = None,
        use_checkpoint = cfg.model_config.use_checkpoint,
        use_fp16 = cfg.model_config.use_fp16,
        num_heads = cfg.model_config.num_heads,
        num_head_channels = cfg.model_config.num_head_channels,
        num_heads_upsample = cfg.model_config.num_heads_upsample,
        use_scale_shift_norm = cfg.model_config.use_scale_shift_norm,
        resblock_updown = cfg.model_config.resblock_updown,
        use_new_attention_order = cfg.model_config.use_new_attention_order,
        num_groups = cfg.model_config.num_groups,
        bottleneck_attention = cfg.model_config.bottleneck_attention,
        additive_skips = cfg.model_config.additive_skips,
        ).to(cfg.device)
        model = JCT(config=None, model=unet, num_timesteps=sample_steps, method=sample_method)
    else:
        raise NotImplementedError(f"{model_name} is not implementde")
    
    return model

# if __name__ == "__main__":
    
    # model = build_model()
