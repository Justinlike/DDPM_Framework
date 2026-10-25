import torch
import torch.nn as nn
#from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
import torch.nn.functional as F
from .difff import MyMaisiEncoder
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi

    
class CTCo(nn.Module):
    def __init__(self, config, num_timesteps=10):
        super().__init__()


        self.num_timesteps = num_timesteps
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_timesteps, beta_start=0.0015, beta_end=0.0155, prediction_type='sample'
        )

        #self.scale = 0.18215

        #self.vae = AutoencoderKlMaisi(**config['autoencoder_def'])
       #self.lant_encoder = MyMaisiEncoder(**config['autoencoder_def'])

        self.unet = DiffusionModelUNetMaisi(**config["diffusion_unet_def"])

        # a = torch.load('/mnt/sda1/lhw/project/CT/checkpoint/autoencoder.pt', weights_only=False)
        # self.vae.load_state_dict(a)

        a = torch.load('/mnt/sda1/lhw/project/CT/checkpoint/diffusion_unet.pt', weights_only=False)
        self.unet.load_state_dict(a, strict=False)
        #self.lant_encoder.init_from_pretrained(self.vae)
        self.loss = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, x, mask=None, **kwargs):
        if mask is not None:
            return self.denosing(x, mask, **kwargs)
        else:
            return self.inference(x, **kwargs)

    def denosing(self, x, mask, **kwargs):

        # x_l = self.lant_encoder(x)[0]   # z_mu, z_sigma
        # x_l = x_l * self.scale
        
        # with torch.no_grad():
        #     x_ml = self.vae.encode(mask)[0]    # z_mu, z_sigma
        #     x_ml = x_ml * self.scale

        t = torch.randint(1, self.num_timesteps, (x.shape[0],), device=x.device, generator=kwargs['generator'])
                        # Sample noise
        noise = torch.randn(
            x.shape,
            device=x.device,
            generator=kwargs['generator']
        )  # [B, 4, h, w]

        #随机融合 当t越来越大gt就会没有，推理是t是最大，然后退gt
        timesteps_expanded = t.view(-1, 1, 1, 1, 1)
        ratio = timesteps_expanded/self.num_timesteps
        #noise = ratio * x_l + (1-ratio) * x_ml
        noise = ratio * x + 2 * (1-ratio) * mask

        # Concat rgb and depth latents
        cat_latents = torch.cat(
            [x, noise], dim=1
        )  # [B, 8, h, w]

        #x = torch.concat([x_ml, x_l], dim=1)   # [B c*2 h w]
        if self.noise_scheduler.config.prediction_type == "previous":
            #直接预测前一时间状态
            ratio = (timesteps_expanded-1)/self.num_timesteps
            target = ratio * x + 2 * (1-ratio) * mask
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = mask
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(mask, noise, t)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        #model_pred, z_tilde = self.unet_wrapper(cat_latents, t)  # [B, 4, h, w]
        #model_pred = self.unet(cat_latents, t).sample
        #model_pred = self.unet(noise, t)
        model_pred = self.unet(cat_latents, t)

        loss = self.loss(model_pred.float(), target.float())
        return {'loss': loss, 'predicted': 0}
    
    def get_model_device(self, model: nn.Module) -> torch.device:
        """从模型参数中推断设备，无参数时返回 CPU"""
        try:
            # 优先从可训练参数中取（若模型有参数）
            return next(model.parameters()).device
        except StopIteration:
            # 无参数时（如纯容器模型），从缓冲区取
            try:
                return next(model.buffers()).device
            except StopIteration:
                # 既无参数也无缓冲区，默认返回 CPU
                return torch.device("cpu")
    @property
    def device(self):
        return self.get_model_device(self.unet)

    def inference(self, x, **kwargs):

        # with torch.no_grad():
        #     x_l = self.lant_encoder(x)[0]
        #     x_l = x_l * self.scale
        
        # Initial depth map (noise)
        # mask_noise = torch.randn(
        #     x_l.shape,
        #     device=x.device,
        #     generator=kwargs['generator']
        # )  # [B, 4, h, w]
        mask_noise = x
        # ratio = (self.num_timesteps-1)/self.num_timesteps
        # mask_noise = ratio*x_l + (1-ratio)*mask_noise  #随机扰动

        #for t in range(self.num_timesteps, 0, -1):
        for t in [9]:
            unet_input = torch.cat(
                [x, mask_noise], dim=1
            )  # this order is important
            #unet_input = mask_noise

            timestep = torch.tensor([t]).repeat(x.shape[0]).to(x.device).long()
            #下一时间点预测
            mask_noise = self.unet(unet_input, timestep)  # [B, 4, h, w]
            mask_noise = torch.clamp(mask_noise, -10.0, 10.0)


        mask_pred = mask_noise
        #print(mask_pred.max(), mask_pred.min(), '[[[[[[[]]]]]]]')
        # mask_latent = mask_pred / self.scale
        # mask = self.vae.decode(mask_latent)

            # clip prediction
       #mask = torch.clamp(mask, -2.0, 4.0)
        #mask = torch.clamp(mask, 0, 1.0)
            # shift to [0, 1]
        #mask = (mask + 1.0) / 2.0
            
        return {'predicted': mask_pred}





class CTCo_diff(nn.Module):
    def __init__(self, config, num_timesteps=1000):
        super().__init__()


        self.num_timesteps = num_timesteps
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_timesteps, beta_start=0.0015, beta_end=0.0155, prediction_type='epsilon'
        )

        # self.scale = 0.18215

        # self.vae = AutoencoderKlMaisi(**config['autoencoder_def'])
        # self.lant_encoder = MyMaisiEncoder(**config['autoencoder_def'])

        self.unet = DiffusionModelUNetMaisi(**config["diffusion_unet_def"])

        # a = torch.load('/mnt/sda1/lhw/project/CT/checkpoint/autoencoder.pt', weights_only=False)
        # self.vae.load_state_dict(a)

        a = torch.load('/mnt/sda1/lhw/project/CT/checkpoint/diffusion_unet.pt', weights_only=False)
        self.unet.load_state_dict(a, strict=False)
        #self.lant_encoder.init_from_pretrained(self.vae)
        #self.loss = nn.L1Loss()
        self.loss = nn.MSELoss()
    
    def forward(self, x, mask=None, **kwargs):
        if mask is not None:
            return self.denosing(x, mask, **kwargs)
        else:
            return self.inference(x, **kwargs)

    def denosing(self, x, mask, **kwargs):

        # x_l = self.lant_encoder(x)[0]   # z_mu, z_sigma
        # x_l = x_l * self.scale
        
        # with torch.no_grad():
        #     x_ml = self.vae.encode(mask)[0]    # z_mu, z_sigma
        #     x_ml = x_ml * self.scale

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device, generator=kwargs['generator'])
                        # Sample noise
        noise = 5 * torch.randn(
            x.shape,
            device=x.device,
            generator=kwargs['generator']
        )  # [B, 4, h, w]   (-5, 5)

        noisy_latents = self.noise_scheduler.add_noise(
            mask, noise, t
        )  # [B, 4, h, w]

        # Concat rgb and depth latents
        # cat_latents = torch.cat(
        #     [x_l, noisy_latents], dim=1
        # )  # [B, 8, h, w]
        cat_latents = noisy_latents + x
        cat_latents = cat_latents.float()

        #随机融合 当t越来越大gt就会没有，推理是t是最大，然后退gt
        # timesteps_expanded = t.view(-1, 1, 1, 1)
        # ratio = timesteps_expanded/self.num_timesteps
        # noise = ratio * x_l + (1-ratio) * x_ml

        # # Concat rgb and depth latents
        # cat_latents = torch.cat(
        #     [x_l, noise], dim=1
        # )  # [B, 8, h, w]

        #x = torch.concat([x_ml, x_l], dim=1)   # [B c*2 h w]
        if self.noise_scheduler.config.prediction_type == "epsilon":
            #直接预测前一时间状态
            target = noise
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = x_ml
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x_ml, noise, t)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        #model_pred, z_tilde = self.unet_wrapper(cat_latents, t)  # [B, 4, h, w]
        #model_pred = self.unet(cat_latents, t).sample
        model_pred = self.unet(cat_latents, t)

        loss = self.loss(model_pred.float(), target.float())

        return {'loss': loss, 'predicted': 0}
    
    def get_model_device(self, model: nn.Module) -> torch.device:
        """从模型参数中推断设备，无参数时返回 CPU"""
        try:
            # 优先从可训练参数中取（若模型有参数）
            return next(model.parameters()).device
        except StopIteration:
            # 无参数时（如纯容器模型），从缓冲区取
            try:
                return next(model.buffers()).device
            except StopIteration:
                # 既无参数也无缓冲区，默认返回 CPU
                return torch.device("cpu")
    @property
    def device(self):
        return self.get_model_device(self.unet)

    def inference(self, x, **kwargs):

        # with torch.no_grad():
        #     x_l = self.lant_encoder(x)[0]
        #     x_l = x_l * self.scale
        
        # Initial depth map (noise)
        mask_noise = 5 * torch.randn(
            x.shape,
            device=x.device,
            generator=kwargs['generator']
        )  # [B, 4, h, w]
        
        num_inference_steps = self.num_timesteps
        self.noise_scheduler.set_timesteps(self.num_timesteps, x.device)
        timesteps = self.noise_scheduler.timesteps
        
        iterable = enumerate(timesteps)

        unet_input = x + mask_noise
        for i, t in iterable:
            # unet_input = torch.cat(
            #     [x_l, mask_noise], dim=1
            # )  # this order is important
            unet_input = x + mask_noise

            # predict the noise residual
            timestep = torch.tensor([t]).repeat(x.shape[0]).to(x.device).long()
            model_pred = self.unet(unet_input, timestep)  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            if num_inference_steps == 1 and self.noise_scheduler.config.prediction_type != "epsilon":
                mask_noise = self.noise_scheduler.step(
                    model_pred, t, mask_noise
                ).pred_original_sample
            else:
                mask_noise = self.noise_scheduler.step(
                    model_pred, t, mask_noise
                ).prev_sample

        mask_pred = mask_noise
        # mask_latent = mask_pred / self.scale
        # mask = self.vae.decode(mask_latent)

            # clip prediction
        #mask = torch.clamp(mask, -2.0, 4.0)
        #mask = torch.clamp(mask, 0, 1.0)
            # shift to [0, 1]
        #mask = (mask + 1.0) / 2.0
            
        return {'predicted': mask_pred}

