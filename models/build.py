import timm
# import segmentation_models_pytorch as smp
from .semi_ct import CTCo, CTCo_diff
# from .Leaf import *
import json
from .jct import JCT
from .jct_noise import JCT_noise
from .jct2d import JCT2D, JCT2D_noise

def build_model(cfg):
    model_name = cfg.model.lower()

    if model_name == "jct":
        with open('/data/jzk/CT/inference.json') as f:
            conf = json.load(f)
        model = JCT(conf)
    elif model_name == 'jct_noise':
        with open('/data/jzk/CT/inference.json') as f:
            conf = json.load(f)
        model = JCT_noise(conf)
    elif model_name == 'jct2d':
        model = JCT2D(cfg)
    elif model_name == 'jct2d_noise':
        model = JCT2D_noise(cfg)

    else:
        raise NotImplementedError(f"{model_name} is not implementde")
    
    return model