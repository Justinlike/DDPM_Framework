import torch
import timm
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, backbone_name, pretrained):
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=[2,3,4]
        )

model = timm.create_model('unet', pretrained=False)
