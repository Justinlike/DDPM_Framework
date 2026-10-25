import torch
import torch.nn as nn
#from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
import torch.nn.functional as F
#from .difff import MyMaisiEncoder
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from .unet import Unet_class, init_weights

    
class JCT2D(nn.Module):
    def __init__(self, config, num_timesteps=50):
        super().__init__()


        self.num_timesteps = num_timesteps
    
        #self.unet = DiffusionModelUNetMaisi(**config["diffusion_unet_def"])
        self.unet = Unet_class(dim=32, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=1, resnet_block_groups=8)
        init_weights(self.unet, 'kaiming')
        # self.unet = UNetModel(config['resolution'], in_channels=1, out_channels=1,
        #                        dropout=0.2,
        #                         num_heads=4,
        #                         resblock_updown=True,
        #                         use_scale_shift_norm=True,
        #                         attention_resolutions = (0, 0, 4, 4),
        #                         channel_mult = (1, 2, 4, 4))

        # a = torch.load('/mnt/sda1/lhw/project/CT/checkpoint/diffusion_unet.pt', weights_only=False)
        # self.unet.load_state_dict(a, strict=False)

        # self.P_std = 0.8
        # self.P_mean = -0.8
        self.P_std = 1.0
        self.P_mean = 0
        self.noise_scale = 1.0
        self.t_eps = 5e-2
        self.method = "heun"

        self.loss = nn.L1Loss()
    
    def forward(self, x, mask=None, **kwargs):
        #print(x.shape, '.....', x.dtype, x.min(), x.max())
        if mask is not None:
            #print(x.shape, '.....', x.dtype, x.min(), x.max(), mask.min(), mask.max())
            return self.predict(x, mask, **kwargs)
        else:
            return self.generate(x, **kwargs)
        
    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def predict(self, x, mask, **kwargs):

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))

        # here is ncct
        e = x * self.noise_scale
        # Sample noise and mask is ccta  # gengerate process is add
        z = t * mask  + (1 - t) * e
        
        #post norm
        #z = (z - z.mean()) / z.std()
        #mask = (mask - mask.mean()) / mask.std()
        #
        v = (mask - z) / (1 - t).clamp_min(self.t_eps)

        #x_pred = self.net(z, t.flatten())
        x_pred = self.unet(z, t.flatten())
        #v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        #预测速度
        v_pred = x_pred

        # l2 loss
        # loss = (v - v_pred) ** 2
        # loss = loss.mean(dim=(1, 2, 3)).mean()
        loss = self.loss(v_pred, v)

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

    @torch.no_grad()
    def generate(self, x, **kwargs):
        # ncct -> ccta
        z = self.noise_scale * x

        timesteps = torch.linspace(0.0, 1.0, self.num_timesteps+1, device=x.device).view(-1, *([1] * z.ndim)).expand(-1, x.size(0), -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError
        

        # ode
        for i in range(self.num_timesteps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1])

        z = z.clamp(-1.0, 1.0)

        return {'predicted': z}
    
    @torch.no_grad()
    def _forward_sample(self, z, t):
        # conditional
        with torch.no_grad():
            x_0 = self.unet(z, t.flatten())
        #v = (x_0 - z) / (1.0 - t).clamp_min(self.t_eps)
        v = x_0

        return v
    
    @torch.no_grad()
    def _euler_step(self, z, t, t_next):
        v_pred = self._forward_sample(z, t)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next):
        v_pred_t = self._forward_sample(z, t)

        z_next_euler = z + (t_next - t) * v_pred_t

        #z_next_euler = z_next_euler.clamp(-1.0, 1.0)
        v_pred_t_next = self._forward_sample(z_next_euler, t_next)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        #z_next = z_next.clamp(-1.0, 1.0)
        return z_next

