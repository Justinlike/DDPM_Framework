import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine noise schedule
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alpha_cumprod = torch.cos(
        ((x/timesteps)+s) / (1+s) * torch.pi
    ) ** 2
    alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
    betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
    #  如何切分，要不要有0？
    return torch.clip(betas, 0.0001, 0.9999)

    

class GaussionDiffusion(nn.Module):
    def __init__(
        self, 
        text_use_bert_cls = False,
        channels = 3,
        timesteps = 1000,
        loss_type = "l1",
        use_dynamic_thers = False,
        dynamic_thers_percetile = 0.9,
    ):
        super().__init__()
        self.channels = channels

        # define beta schedule
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        alphas_cumprod_prev = F.pad()

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffers help fuc, cast torch.float64 to torch.float32

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculate for diffusion q(x_t|x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # wait tor check
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn,x, t, clip_denoised: bool, y=None, res=None,hint = None,cond_scale=1.):
        if hint == None:
            x_recon = self.predict_start_from_noise(
                        x, t=t, noise=denoise_fn(x, t, y=y,res=res))
        else:
            x_recon = self.predict_start_from_noise(
            x, t=t, noise=denoise_fn(x, t, y=y, res=res,hint = hint))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, denoise_fn , x, t, y=None, res= None,hint = None,cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn,
            x=x, t=t, clip_denoised=clip_denoised, y=y, res=res,hint = hint,cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                        *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, denoise_fn, z, y=None, res=None,cond_scale=1.,hint = None):
        device = self.betas.device

        b = z.shape[0]
        img = default(z, lambda: torch.randn_like(z , device= device))

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(denoise_fn, img, torch.full(
                (b,), i, device=device, dtype=torch.long), y=y, res=res,cond_scale=cond_scale,hint= hint)

        return img

    @torch.inference_mode()
    def sample(self, denoise_fn, z, y=None, res=None,cond_scale=1.,hint = None, strategy='ddpm', eta=0.0, ddim_steps= 100):
        if strategy == 'ddpm':
            return self.p_sample_loop(
                        denoise_fn, z, y, res, cond_scale, hint
                    )
        elif strategy == 'ddim':
            return self.p_sample_loop_ddim(
                denoise_fn, z, y, res, cond_scale, hint, eta, ddim_steps
            )
        else:
            raise NotImplementedError


    @torch.inference_mode()
    def ddim_sample(self, denoise_fn, x, x_cond, t, t_prev, y=None, hint=None, cond_scale=1., clip_denoised=False):
        b, *_, device = *x.shape, x.device

        # Get predicted x0
        if hint is None:
            x_recon = denoise_fn(torch.cat((x, x_cond), dim=1), t, y=y)
        else:
            x_recon = denoise_fn(torch.cat((x, x_cond), dim=1), t, y=y, hint=hint)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))
            x_recon = x_recon.clamp(-s, s) / s

        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1)
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_alpha_prev = alpha_prev.sqrt()

        # compute direction pointing to x_t
        eps = (x - sqrt_alpha_t * x_recon) / (1 - alpha_t).sqrt()

        # deterministic DDIM update
        x_prev = sqrt_alpha_prev * x_recon + (1 - alpha_prev).sqrt() * eps
        return x_prev


    @torch.inference_mode()
    def p_sample_loop_ddim(self, denoise_fn, z, y=None, res=None, cond_scale=1., hint=None, eta=0.0, ddim_steps=100):
        device = self.betas.device
        b = z.shape[0]
        img = default(z, lambda: torch.randn_like(z, device=device))

        # DDIM time schedule
        times = torch.linspace(0, self.num_timesteps - 1, steps=ddim_steps).long().flip(0).to(device)


        for i in tqdm(range(ddim_steps), desc="DDIM Sampling", total=ddim_steps):
            t = times[i]
            t_prev = times[i + 1] if i < ddim_steps - 1 else torch.tensor(0, device=device)
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            t_prev_batch = torch.full((b,), t_prev, device=device, dtype=torch.long)

            img = self.p_sample_ddim(
                denoise_fn, img, t_batch, t_prev_batch,
                y=y, res=res, hint=hint, cond_scale=cond_scale,
                eta=eta, clip_denoised=True
            )

        return img

    @torch.inference_mode()
    def p_sample_ddim(self, denoise_fn, x, t, t_prev, y=None, res=None, hint=None, cond_scale=1., clip_denoised=True, eta=0.0):
        # Predict noise (ε_theta)
        if hint is None:
            noise_pred = denoise_fn(x, t, y=y, res=res)
        else:
            noise_pred = denoise_fn(x, t, y=y, res=res, hint=hint)

        # Predict x0
        x0 = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x0, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x0.ndim - 1)))
            x0 = x0.clamp(-s, s) / s

        # DDIM formula
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
   
        alpha_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1)
        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        
        pred_dir = (1 - alpha_prev).sqrt() * noise_pred
        x_prev = alpha_prev.sqrt() * x0 + pred_dir

        if eta > 0:
            noise = torch.randn_like(x)
            x_prev = x_prev + sigma * noise

        return x_prev

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, denoise_fn,x_start, t, y=None, res=None,noise=None, hint = None,**kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if is_list_str(y):
            y = bert_embed(
                tokenize(y), return_cls_repr=self.text_use_bert_cls)
            y = y.to(device)
        if hint == None:
            x_recon = denoise_fn(x_noisy, t, y=y,res=res)
        else:
            x_recon = denoise_fn(x_noisy, t, y=y, res=res,hint = hint)
        # time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss
