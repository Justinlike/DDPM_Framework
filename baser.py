import torch
import torchio as tio
from tools import *
from einops import rearrange

class Baser():
    def __init__(self, args, model, loss_fn, **kwaegs):
        self.args = args
        self.model = model
        self.criterion = loss_fn

    def forward(self, samples, require_grad=True, **kwargs):
        # imgs = samples['imgs'][tio.DATA].cuda(non_blocking=True)
        # #imgs = imgs * 2.0 - 1.0
        # labels=samples['labels'][tio.DATA].cuda(non_blocking=True)
        #labels = labels * 2.0 - 1.0
        try:
            imgs = samples['imgs'].cuda(non_blocking=True) # 遇到 TorchIO Image 对象报错，进入except
            #imgs = (imgs/255) * 2.0 - 1.0
            imgs = imgs * 2.0 - 1.0
            labels=samples['labels'].cuda(non_blocking=True)
            #imgs = (labels/255) * 2.0 - 1.0
            labels = labels * 2.0 - 1.0
            #print(imgs.shape, imgs.max(), imgs.min())
            #print(imgs.max(), imgs.min(), labels.max(), labels.min())
        except:
            imgs = samples['imgs'][tio.DATA].unsqueeze(0).cuda(non_blocking=True)
            #print(imgs.shape, '/?????????', imgs.max(), imgs.min())
            # 去掉 rearange 因为导致了维度问题
            # imgs = rearrange(imgs, 'b c h w z -> (b z) c h w') 
            imgs = imgs * 2.0 - 1.0
            #imgs = imgs * 2.0 - 1.0
            labels=samples['labels'][tio.DATA].unsqueeze(0).cuda(non_blocking=True)
            labels = rearrange(labels, 'b c h w z -> (b z) c h w')
            labels = labels * 2.0 - 1.0

            #return {'predicted': imgs.clamp(-10,10)}

        # autocast 让内部符合条件的算子自动用低精度
        # CUDA: fp16/bfloat16，CPU: bfloat16
        with torch.autocast(imgs.device.type):
            if require_grad:
                output = self.model(imgs, labels, **kwargs)
            else:
                with torch.no_grad():
                    #output = self.model(imgs, len_list=samples['imgs_len'])
                    output = self.model(imgs, **kwargs)

        return output
    
    def compute_loss(self, output, labels):
        if self.args.ctiterion == "in":
            return output["loss"]

        return self.criterion(output["predicted"], labels.to(output["predicted"].dtype))

    def postprocess(self, batch, output):
        # predicteds = F.softmax(predicteds, dim=-1)
        # predicteds = torch.argmax(predicteds, dim=-1).detach().cpu()
        #predicteds = F.logsigmoid(predicteds).exp().detach().cpu()
        predicteds = output['predicted'].detach().cpu()
        #predicteds = 255 * ((predicteds + 1) / 2)   # [0,255]
        #targets = batch['labels'][tio.DATA].detach().cpu()
        try:
            targets = batch['labels'].detach().cpu()
        except:
            targets = batch['labels'][tio.DATA].unsqueeze(0).detach().cpu()
            # 修复维度问题
            # predicteds = rearrange(predicteds, '(b z) c h w -> b c h w z', b=1)
            #targets = rearrange(targets, '(b z) c h w -> b c h w z', b=1)
            #print(predicteds.shape, targets.shape, targets.max(), targets.min())
        #targets = 255 * targets   # [0,255]
        targets = (2 * targets) - 1
        # predicteds = torch.where(torch.mean(predicteds, dim=1, keepdim=True) > 0.5, 1, 0).long()
        # targets = torch.mean(targets, dim=1, keepdim=True).long()
        # predicteds = torch.where(torch.mean(predicteds, dim=1, keepdim=True) > 0.5, 1, 0).long()
        # targets = torch.mean(targets, dim=1, keepdim=True).long()

        return predicteds, targets