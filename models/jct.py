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


class JCT(nn.Module):
    def __init__(self, config, num_timesteps=50):
        super().__init__()

        self.num_timesteps = num_timesteps
    
        self.unet = DiffusionModelUNetMaisi(**config["diffusion_unet_def"])

        # a = torch.load('/mnt/sda1/lhw/project/CT/checkpoint/diffusion_unet.pt', weights_only=False)
        # self.unet.load_state_dict(a, strict=False)

        # self.P_std = 0.8
        # self.P_mean = -0.8
        self.P_std = 1.0
        self.P_mean = 0
        self.noise_scale = 1.0
        self.t_eps = 5e-2
        self.method = "heun"

    
    def forward(self, x, mask=None, **kwargs):
        if mask is not None:
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
        # t CCTA + (1-t) NCCT
        z = t * mask  + (1 - t) * e
        
        #post norm
        #z = (z - z.mean()) / z.std()
        #mask = (mask - mask.mean()) / mask.std()
        #
        v = (mask - z) / (1 - t).clamp_min(self.t_eps)

        #x_pred = self.net(z, t.flatten())
        # print("x shape:", x.shape)
        # print("mask shape:", mask.shape)
        # print("z shape before UNet:", z.shape)
        # print("t shape:", t.shape)
        x_pred = self.unet(z, t.flatten())
        # v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        #预测速度
        v_pred = x_pred

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3, 4)).mean()

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
        # # 调试信息
        # print("=" * 60)
        # print("generate input x.shape =", x.shape)

        # ncct -> ccta
        # z = torch.randn_like(x, device=x.device)
        # mu_x = x.mean()
        # sigma_x = x.std()
        # #噪声分布调整至ncct
        # z = self.noise_scale * (mu_x + sigma_x * z)
        z = self.noise_scale * x

        # ### 
        # # 调试信息
        # print("z.shape =", z.shape)
        # print("z.ndim =", z.ndim)
        # print("x.size(0) =", x.size(0))
        # print("=" * 60)

        # raw_t = torch.linspace(
        #     0.0,
        #     1.0,
        #     self.num_timesteps + 1,
        #     device=x.device
        # )
        # print("raw timestep shape =", raw_t.shape)

        # tmp_t = raw_t.view(
        #     -1,
        #     *([1] * z.ndim)
        # )
        # print("after view shape =", tmp_t.shape)
        # ###

        timesteps = torch.linspace(0.0, 1.0, self.num_timesteps+1, device=x.device).view(-1, *([1] * z.ndim)).expand(-1, x.size(0), -1, -1, -1, -1)

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
        v_pred_t_next = self._forward_sample(z_next_euler, t_next)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

