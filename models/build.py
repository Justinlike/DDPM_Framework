import timm
# import segmentation_models_pytorch as smp
from .semi_ct import CTCo, CTCo_diff
# from .Leaf import *
import json
from .jct import JCT
from .jct_noise import JCT_noise
from .jct2d import JCT2D, JCT2D_noise
from .unet import UNet, WUNet
from omegaconf import OmegaConf

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
        model_kwargs = OmegaConf.to_container(
            cfg.model_config,
            resolve=True,
        )
        model_kwargs.update(
            num_classes=None
        )
        unet = UNet(**model_kwargs)

        model = JCT(config=None, model=unet, num_timesteps=sample_steps, method=sample_method)
    elif model_name == "jct_wunet":
        model_kwargs = OmegaConf.to_container(
                    cfg.model_config,
                    resolve=True,
                )
        unet = WUNet(**model_kwargs)
        model = JCT(config=None, model=unet, num_timesteps=sample_steps, method=sample_method)
    else:
        raise NotImplementedError(f"{model_name} is not implementde")
    
    return model

# if __name__ == "__main__":
    
    # model = build_model()
