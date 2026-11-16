from pathlib import Path
import torch
from omegaconf import OmegaConf
from models.vqgan_3d.vqgan import VQGAN
from train_vqgan import _make_dataloaders
import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def load_vqgan(checkpoint_path, device=None, config_path=None):
    """加载自己的可信 Lightning checkpoint；裸 state_dict 需提供训练配置。

    优先使用 checkpoint 中保存的 cfg，不使用当前 YAML 覆盖训练压缩比。
    """
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    cfg = checkpoint.get("hyper_parameters", {}).get("cfg")
    if cfg is None:
        if config_path is None:
            raise ValueError("checkpoint 中没有 cfg，请提供训练时的 config_path")
        cfg = OmegaConf.load(config_path)
    elif isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    model = VQGAN(cfg)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    model.codebook._need_init = False
    model.eval().requires_grad_(False)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


@torch.inference_mode()
def generate_latent(model, image, *, layout="DHW", output_layout="DHW", quantize=False):
    """接收 tensor/ndarray，支持三维、[C,...] 或 [B,C,...] 输入。

    layout="DHW": 已有训练 DataLoader 输出可直接传入。
    layout="HWD": 原始图像，内部转换为训练采用的 DHW。
    layout="native": 自定义训练流程的原生轴序，不做转换；输出也须 native。
    quantize=False: 连续 pre_vq_conv 特征；True: 最近码本向量（不是整数索引）。
    不进行归一化、resize 或额外 latent 缩放，返回值保留 batch/channel 轴。
    """
    if layout not in {"DHW", "HWD", "native"}:
        raise ValueError("layout 必须是 DHW、HWD 或 native")
    if output_layout not in {"DHW", "HWD", "native"}:
        raise ValueError("output_layout 必须是 DHW、HWD 或 native")
    if (layout == "native") != (output_layout == "native"):
        raise ValueError("原生轴序输入须同时指定 output_layout='native'")
    x = torch.as_tensor(image)
    if x.ndim == 3:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 4:
        x = x.unsqueeze(0)
    if x.ndim != 5:
        raise ValueError("输入需为 [S1,S2,S3]、[C,S1,S2,S3] 或 [B,C,S1,S2,S3]")
    if layout == "HWD":
        x = x.permute(0, 1, 4, 2, 3)
    if x.shape[1] != model.cfg.dataset.image_channels:
        raise ValueError("输入通道数与训练配置不一致")
    factors = tuple(int(v) for v in model.cfg.model.downsample)
    if any(s % f for s, f in zip(x.shape[2:], factors)):
        raise ValueError(f"输入空间尺寸 {tuple(x.shape[2:])} 必须被训练压缩比 {factors} 整除")
    parameter = next(model.encoder.parameters())
    x = x.to(device=parameter.device, dtype=parameter.dtype).contiguous()
    model.eval()
    latent = model.encode(x, quantize=False)
    if quantize:
        # 分块查找最近码本向量，避免原 Codebook.forward 构建巨大的 one-hot。
        codes = model.codebook.embeddings.float()
        flat = latent.movedim(1, -1).reshape(-1, latent.shape[1]).float()
        indices = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            code_norm = codes.square().sum(1)
            for chunk in flat.split(4096):
                distances = chunk.square().sum(1, keepdim=True) - 2 * chunk @ codes.T + code_norm
                indices.append(distances.argmin(1))
            latent = torch.nn.functional.embedding(torch.cat(indices), codes).reshape(
                latent.shape[0], *latent.shape[2:], latent.shape[1]
            ).movedim(-1, 1).contiguous()
    if output_layout == "HWD":
        latent = latent.permute(0, 1, 3, 4, 2).contiguous()
    return latent

if __name__ == "__main__":
    cfg = OmegaConf.load("/data/jzk/DDPM_Framework/conf/config-vqgan-infer.yaml")
    train_dataloader, val_dataloader = _make_dataloaders(cfg)

    model = load_vqgan("/data/jzk/DDPM_Framework/output/vq_gan/CT3D/single_gpu2_gan_train/lightning_logs/version_1/checkpoints/latest_checkpoint-v1.ckpt", device="cuda:0")
    # print(model)
    model.eval()
    # /data/jzk/DDPM_Framework/output/vq_gan/CT3D/single_gpu2_gan_train/lightning_logs/version_1/checkpoints/epoch=2-step=57000-train/recon_loss=0.04.ckpt
    # # image: [512,512,64]，使用与训练一致的归一化
    # for img in train_dataloader:
    #     latent = generate_latent(
    #         model, img,
    #         layout="HWD",
    #         output_layout="HWD",
    #         quantize=False,  # True 则返回量化后的码本向量
    #     )
    #     print("latent shape:", latent.shape)

    # batch = next(iter(train_dataloader))
    # print(batch.keys())  # 先查看实际字段名    
    parameter = next(model.encoder.parameters())
    psnr_list = []

    for batch in val_dataloader:
        img = batch["data"].to(
            device=parameter.device,  # 跟随模型，当前是 cuda:2
            dtype=parameter.dtype,
            non_blocking=True,
        )
        print("ID", batch["id"], "img shape:", img.shape)  # [B, C, D, H, W]
        reference = img.cpu().float() # [B, C, D, H, W]

        # volume = (
        #     batch["data"][0, 0]
        #     .permute(1, 2, 0)  # [D,H,W] → [H,W,D]
        #     .detach()
        #     .cpu()
        #     .numpy()
        # )
        # affine = batch["affine"][0].detach().cpu().numpy()
        # nib.save(
        #     nib.Nifti1Image(volume, affine),
        #     "input.nii.gz",
        # )
        
        with torch.inference_mode():
            z_quantized, indices = model.encode(
                img,
                quantize=True,
                include_embeddings=True,
            )
            print("z_quantized shape:", z_quantized.shape)
            # print("indices shape:", indices.shape)

            # 整幅 latent 和重建结果放在 CPU，每次只搬一个块到 GPU。
            reconstruction = torch.empty(img.shape, dtype=torch.float32, device="cpu")
            latent = z_quantized.cpu()
            del z_quantized, indices, img

            core, halo = 32, 8
            d_scale, h_scale, w_scale = 2, 4, 4
            depth, height, width = latent.shape[-3:]

            for h0 in range(0, height, core):
                h1 = min(h0 + core, height)
                ph0 = max(0, h0 - halo)
                ph1 = min(height, h1 + halo)

                for w0 in range(0, width, core):
                    w1 = min(w0 + core, width)
                    pw0 = max(0, w0 - halo)
                    pw1 = min(width, w1 + halo)

                    tile = latent[..., ph0:ph1, pw0:pw1].contiguous()
                    tile = tile.to(device=parameter.device)

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        decoded = model.decoder(model.post_vq_conv(tile))

                    expected_shape = (
                        depth * d_scale,
                        (ph1 - ph0) * h_scale,
                        (pw1 - pw0) * w_scale,
                    )
                    if tuple(decoded.shape[-3:]) != expected_shape:
                        raise RuntimeError(
                            f"解码块尺寸为 {tuple(decoded.shape[-3:])}，"
                            f"预期为 {expected_shape}，请检查 decoder 的空间放大倍率。"
                        )

                    decoded_core = decoded[
                        ...,
                        (h0 - ph0) * h_scale:(h1 - ph0) * h_scale,
                        (w0 - pw0) * w_scale:(w1 - pw0) * w_scale,
                    ].cpu().float()

                    reconstruction[
                        ...,
                        h0 * h_scale:h1 * h_scale,
                        w0 * w_scale:w1 * w_scale,
                    ] = decoded_core

                    del tile, decoded

            print("reconstruction shape:", reconstruction.shape)
            # volume = (
            #     reconstruction[0, 0]
            #     .permute(1, 2, 0)  # [D,H,W] → [H,W,D]
            #     .detach()
            #     .cpu()
            #     .numpy()
            # )
            # nib.save(nib.Nifti1Image(volume, affine),"reconstruction.nii.gz",)
            psnr = peak_signal_noise_ratio(reference.numpy()[0, 0], reconstruction.numpy()[0, 0])
            psnr_list.append(psnr)
            print(f"PSNR: {psnr:.4f}")
        # with torch.inference_mode():
        #     z_quantized, indices = model.encode(
        #         img,
        #         quantize=True,
        #         include_embeddings=True,
        #     )
        #     print("z_quantized shape:", z_quantized.shape)  # [B, C, D', H', W']
        #     print("indices shape:", indices.shape)  # [B, D', H', W']

        #     with torch.autocast(device_type="cuda", dtype=torch.float16):
        #         # 示例。若你第 119 行用的是其他输入变换，保留实际写法。
        #         reconstruction = model.decoder(model.post_vq_conv(z_quantized))

        #     # 先移到 CPU，再转 float32，避免在 GPU 上额外创建一份 FP32 输出。
        #     reconstruction = reconstruction.cpu().float()
            
        #     # # 方法一：解码量化后的向量
        #     # reconstruction = model.decoder(
        #     #     model.post_vq_conv(z_quantized)
        #     # )
        #     # print("reconstruction shape:", reconstruction.shape)  # [B, C, D, H, W]
    
        #     # # 方法二：解码码本索引，结果与方法一对应
        #     # reconstruction = model.decode(indices)
        #     # print("reconstruction shape:", reconstruction.shape)  # [B, C, D, H, W]

        # latent = generate_latent(
        #     model, img,
        #     layout="HWD",
        #     output_layout="HWD",
        #     quantize=False,  # True 则返回量化后的码本向量
        # )
        # print("latent shape:", latent.shape)
        # break
    for batch in train_dataloader:
        img = batch["data"].to(
            device=parameter.device,  # 跟随模型，当前是 cuda:2
            dtype=parameter.dtype,
            non_blocking=True,
        )
        print("ID", batch["id"], "img shape:", img.shape)  # [B, C, D, H, W]
        reference = img.cpu().float() # [B, C, D, H, W]

        # volume = (
        #     batch["data"][0, 0]
        #     .permute(1, 2, 0)  # [D,H,W] → [H,W,D]
        #     .detach()
        #     .cpu()
        #     .numpy()
        # )
        # affine = batch["affine"][0].detach().cpu().numpy()
        # nib.save(
        #     nib.Nifti1Image(volume, affine),
        #     "input.nii.gz",
        # )
        
        with torch.inference_mode():
            z_quantized, indices = model.encode(
                img,
                quantize=True,
                include_embeddings=True,
            )
            print("z_quantized shape:", z_quantized.shape)
            # print("indices shape:", indices.shape)

            # 整幅 latent 和重建结果放在 CPU，每次只搬一个块到 GPU。
            reconstruction = torch.empty(img.shape, dtype=torch.float32, device="cpu")
            latent = z_quantized.cpu()
            del z_quantized, indices, img

            core, halo = 32, 8
            d_scale, h_scale, w_scale = 2, 4, 4
            depth, height, width = latent.shape[-3:]

            for h0 in range(0, height, core):
                h1 = min(h0 + core, height)
                ph0 = max(0, h0 - halo)
                ph1 = min(height, h1 + halo)

                for w0 in range(0, width, core):
                    w1 = min(w0 + core, width)
                    pw0 = max(0, w0 - halo)
                    pw1 = min(width, w1 + halo)

                    tile = latent[..., ph0:ph1, pw0:pw1].contiguous()
                    tile = tile.to(device=parameter.device)

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        decoded = model.decoder(model.post_vq_conv(tile))

                    expected_shape = (
                        depth * d_scale,
                        (ph1 - ph0) * h_scale,
                        (pw1 - pw0) * w_scale,
                    )
                    if tuple(decoded.shape[-3:]) != expected_shape:
                        raise RuntimeError(
                            f"解码块尺寸为 {tuple(decoded.shape[-3:])}，"
                            f"预期为 {expected_shape}，请检查 decoder 的空间放大倍率。"
                        )

                    decoded_core = decoded[
                        ...,
                        (h0 - ph0) * h_scale:(h1 - ph0) * h_scale,
                        (w0 - pw0) * w_scale:(w1 - pw0) * w_scale,
                    ].cpu().float()

                    reconstruction[
                        ...,
                        h0 * h_scale:h1 * h_scale,
                        w0 * w_scale:w1 * w_scale,
                    ] = decoded_core

                    del tile, decoded

            print("reconstruction shape:", reconstruction.shape)
            # volume = (
            #     reconstruction[0, 0]
            #     .permute(1, 2, 0)  # [D,H,W] → [H,W,D]
            #     .detach()
            #     .cpu()
            #     .numpy()
            # )
            # nib.save(nib.Nifti1Image(volume, affine),"reconstruction.nii.gz",)
            psnr = peak_signal_noise_ratio(reference.numpy()[0, 0], reconstruction.numpy()[0, 0])
            psnr_list.append(psnr)
            print(f"PSNR: {psnr:.4f}")
    print(f"Average PSNR: {np.mean(psnr_list):.4f}")