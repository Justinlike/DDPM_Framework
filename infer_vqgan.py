from pathlib import Path
import torch
from omegaconf import OmegaConf
from models.vqgan_3d.vqgan import VQGAN
from train_vqgan import _make_dataloaders, _make_dataset
import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

def load_vqgan(checkpoint_path, device=None, config_path=None):
    """
    加载自己的可信 Lightning checkpoint；裸 state_dict 需提供训练配置。

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

def got_latent(
        dataloader,
        model,
        output_dir, 
        split="test", 
        rgb="ncct",
        quantize=False, 
        save_latent=True, 
        eval=False,
    ):
    """将 dataloader 中的图像编码为 latent 并保存为 NIfTI 文件。

    output_dir: 输出目录，按 batch["id"] 命名。
    layout: 输入图像轴序，见 generate_latent。
    quantize: if true, 保存量化后的 latent。
    save_latent: 是否保存 latent。
    eval: 是否进行评估。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / split / rgb

    parameter = next(model.encoder.parameters())
    psnr_list, ssim_list = [], []
    for batch in dataloader:
        img = batch["data"].to(
            device=parameter.device,  # 跟随模型，当前是 cuda:1
            dtype=parameter.dtype,
            non_blocking=True,
        )
        print("ID", batch["id"][0], "img shape:", img.shape)  # [B, C, D, H, W]
        reference = img.cpu().float() # [B, C, D, H, W]
        affine = batch["affine"][0].detach().cpu().numpy()

        with torch.inference_mode():
            ##############################
            # 编码为 latent(after quentized)
            ##############################
            latent = model.encode(
                img,
                quantize=False,
            )
            del img
            print("latent shape:", latent.shape)

            # save latent
            if save_latent:
                # # nib version
                # # 现在顺序为 [B, C, D, H, W]，保存为 NIfTI 时需要 permute 为 [B, D, H, W, C]，再保存为 NIfTI。
                # array = (
                #     latent[0].detach().cpu().float()
                #     .permute(2, 3, 1, 0).contiguous().numpy()
                # )  # [C,D,H,W] -> [H,W,D,C]
                # nib.save(
                #     nib.Nifti1Image(array, affine=np.eye(4)),
                #     save_path / f"{batch['id'][0]}.nii.gz",
                # )

                # torch version
                torch.save(
                    {
                        "latent": latent[0].detach().cpu().float(),
                        "affine": affine,
                    },
                    save_path / f"{batch['id'][0]}.pt",
                )

            ##############################
            # 解码 只有评估时需要解码
            ##############################
            if eval:
                # 整图解码：输入须与 post_vq_conv 的设备和精度一致。
                decode_parameter = next(model.post_vq_conv.parameters())
                latent = latent.to(
                    device=decode_parameter.device, dtype=decode_parameter.dtype,
                )

                # 直接inference mode无法使用，因为会超一点显存，转为fp16来做
                # with torch.inference_mode():
                with torch.autocast(
                    device_type=decode_parameter.device.type,
                    dtype=torch.float16,
                    enabled=decode_parameter.device.type == "cuda",
                ):
                    reconstruction = model.decode(latent, quantize=True)
                del latent

                # 先移到 CPU，再转 float32，避免在 GPU 上额外创建一份 FP32 输出。
                reconstruction = reconstruction.cpu().float()
                print("reconstruction shape:", reconstruction.shape)
                volume = reconstruction[0, 0].permute(1, 2, 0).numpy()
                nib.save(nib.Nifti1Image(volume, affine), "reconstruction.nii.gz")

                psnr = peak_signal_noise_ratio(
                    reference.numpy()[0, 0], 
                    reconstruction.numpy()[0, 0],
                    data_range=2.0
                )
                ssim = structural_similarity(
                    reference.numpy()[0, 0],
                    reconstruction.numpy()[0, 0],
                    data_range=2.0,
                )
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
                del reconstruction, reference

    print(f"Average PSNR: {np.mean(psnr_list):.4f}") if eval else None
    print(f"Average SSIM: {np.mean(ssim_list):.4f}") if eval else None

if __name__ == "__main__":
    cfg = OmegaConf.load("/data/jzk/DDPM_Framework/conf/config-vqgan-infer.yaml")
    # train_dataloader, val_dataloader = _make_dataloaders(cfg)
    train_ncct, train_ccta, val_ncct, val_ccta = _make_dataset(
        cfg, cfg.dataset.train_split
    )
    loader_kwargs = {
        "batch_size": cfg.model.batch_size,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
    }

    train_loader_ncct = DataLoader(
        train_ncct,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )
    train_loader_ccta = DataLoader(
        train_ccta,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )
    val_loader_ncct = DataLoader(
        val_ncct,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )
    val_loader_ccta = DataLoader(
        val_ccta,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )

    # 配置1
    path1 = "/data/jzk/DDPM_Framework/output/vq_gan/CT3D/single_gpu2_gan_train_32/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt"
    # /data/jzk/DDPM_Framework/output/vq_gan/CT3D/single_gpu2_gan_train_32/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt
    #   embedding_dim: 256 # 256
    #   n_codes: 2048
    #   n_hiddens: 32 # 240
    #   lr: 3.0e-4
    #   downsample: [2, 4, 4] # D, H, W [2, 4, 4]
    #   disc_channels: 64
    #   disc_layers: 3

    # 配置2 
    path2 = "/data/jzk/DDPM_Framework/output/vq_gan/CT3D/single_gan_train_Embedding8/lightning_logs/version_2/checkpoints/latest_checkpoint.ckpt"
    #   embedding_dim: 8
    #   n_codes: 2048
    #   n_hiddens: 64 # 240 64 8
    #   lr: 3.0e-4
    #   downsample: [2, 4, 4] # D, H, W

    model = load_vqgan(path2, device="cuda:0")
    # print(model)
    model.eval()

    # image: [512,512,64]，使用与训练一致的归一化
    parameter = next(model.encoder.parameters())
    psnr_list = []
    ssim_list = []
    
    print(len(train_loader_ncct))
    print(len(train_loader_ccta))

    # got_latent(
    #     train_loader_ncct,
    #     model,
    #     output_dir="/data/jzk/CT/data_522/",
    #     split="train",
    #     rgb="ncct_latents",
    #     save_latent=True,
    #     eval=False,
    # )
    
    # got_latent(
    #     train_loader_ccta,
    #     model,
    #     output_dir="/data/jzk/CT/data_522/",
    #     split="train",
    #     rgb="ccta_latents",
    #     save_latent=True,
    #     eval=False,
    # )

    got_latent(
        val_loader_ncct,
        model,
        output_dir="/data/jzk/CT/data_522/",
        split="test",
        rgb="ncct_latents",
        save_latent=False,
        eval=True,
    )

    got_latent(
        val_loader_ccta,
        model,
        output_dir="/data/jzk/CT/data_522/",
        split="test",
        rgb="ccta_latents",
        save_latent=False,
        eval=True,
    )

    # for batch in val_dataloader:
    #     img = batch["data"].to(
    #         device=parameter.device,  # 跟随模型，当前是 cuda:1
    #         dtype=parameter.dtype,
    #         non_blocking=True,
    #     )
    #     print("ID", batch["id"], "img shape:", img.shape)  # [B, C, D, H, W]
    #     reference = img.cpu().float() # [B, C, D, H, W]

    #     affine = batch["affine"][0].detach().cpu().numpy()
    #     # nib.save(
    #     #     nib.Nifti1Image(volume, affine),
    #     #     f"input.nii.gz",
    #     # )
        
    #     with torch.inference_mode():

    #         z_quantized, indices = model.encode(
    #             img,
    #             quantize=True,
    #             include_embeddings=True,
    #         )
    #         print("z_quantized shape:", z_quantized.shape)
    #         print("indices shape:", indices.shape)
    #     break
    
    #         # nib.save(
    #         #     nib.Nifti1Image(z_quantized.detach().cpu().numpy(), ),
    #         #     f"/data/jzk/CT/data_522/test/ncct_latents/{batch['id'][0]}.nii.gz",
    #         # )
    
    #         # 整图解码：输入须与 post_vq_conv 的设备和精度一致。
    #         decode_parameter = next(model.post_vq_conv.parameters())
    #         latent = z_quantized.to(
    #             device=decode_parameter.device, dtype=decode_parameter.dtype,
    #         )
    #         del z_quantized, indices, img

    #         with torch.autocast(
    #             device_type=decode_parameter.device.type,
    #             dtype=torch.float16,
    #             enabled=decode_parameter.device.type == "cuda",
    #         ):
    #             reconstruction = model.decoder(model.post_vq_conv(latent))
    #         del latent

    #         # 先移到 CPU，再转 float32，避免在 GPU 上额外创建一份 FP32 输出。
    #         reconstruction = reconstruction.cpu().float()
    #         print("reconstruction shape:", reconstruction.shape)
    #         volume = reconstruction[0, 0].permute(1, 2, 0).numpy()
    #         nib.save(nib.Nifti1Image(volume, affine), "reconstruction.nii.gz")

    #         psnr = peak_signal_noise_ratio(
    #             reference.numpy()[0, 0], 
    #             reconstruction.numpy()[0, 0],
    #             data_range=2.0
    #         )
    #         ssim = structural_similarity(
    #             reference.numpy()[0, 0],
    #             reconstruction.numpy()[0, 0],
    #             data_range=2.0,
    #         )
    #         psnr_list.append(psnr)
    #         ssim_list.append(ssim)
    #         print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")

    #     # break
    # print(f"Average PSNR: {np.mean(psnr_list):.4f}")
    # print(f"Average SSIM: {np.mean(ssim_list):.4f}")
    
    # print("in train_dataloader")
    # for batch in train_dataloader:
    #     # img = batch["data"].to(
    #     #     device=parameter.device,  # 跟随模型，当前是 cuda:2
    #     #     dtype=parameter.dtype,
    #     #     non_blocking=True,
    #     # )
    #     # # print("ID", batch["id"], "img shape:", img.shape)  # [B, C, D, H, W]
    #     # reference = img.cpu().float() # [B, C, D, H, W]

    #     # volume = (
    #     #     batch["data"][0, 0]
    #     #     .permute(1, 2, 0)  # [D,H,W] → [H,W,D]
    #     #     .detach()
    #     #     .cpu()
    #     #     .numpy()
    #     # )
    #     # affine = batch["affine"][0].detach().cpu().numpy()
    #     # nib.save(
    #     #     nib.Nifti1Image(volume, affine),
    #     #     "input.nii.gz",
    #     # )
        
    #     with torch.inference_mode():
    #         img = batch["data"].to(
    #             device=parameter.device,  # 跟随模型，当前是 cuda:1
    #             dtype=parameter.dtype,
    #             non_blocking=True,
    #         )
    #         print("ID", batch["id"], "img shape:", img.shape)  # [B, C, D, H, W]
    #         reference = img.cpu().float() # [B, C, D, H, W]
    
    #         volume = (
    #             batch["data"][0, 0]
    #             .permute(1, 2, 0)  # [D,H,W] → [H,W,D]
    #             .detach()
    #             .cpu()
    #             .numpy()
    #         )
    #         # affine = batch["affine"][0].detach().cpu().numpy()
    #         # nib.save(
    #         #     nib.Nifti1Image(volume, affine),
    #         #     f"/data/jzk/CT/data_522/train/ncct_latents/{batch['id'][0]}.nii.gz",
    #         # )
            
    #         with torch.inference_mode():
    
    #             z_quantized, indices = model.encode(
    #                 img,
    #                 quantize=True,
    #                 include_embeddings=True,
    #             )
    #             print("z_quantized shape:", z_quantized.shape)

    #             nib.save(
    #                 nib.Nifti1Image(z_quantized.detach().cpu().numpy(), ),
    #                 f"/data/jzk/CT/data_522/train/ncct_latents/{batch['id'][0]}.nii.gz",
    #             )

    #             # print("indices shape:", indices.shape)
        
    #             # # 整图解码：输入须与 post_vq_conv 的设备和精度一致。
    #             # decode_parameter = next(model.post_vq_conv.parameters())
    #             # latent = z_quantized.to(
    #             #     device=decode_parameter.device, dtype=decode_parameter.dtype,
    #             # )
    #             # del z_quantized, indices, img
    
    #         #     with torch.autocast(
    #         #         device_type=decode_parameter.device.type,
    #         #         dtype=torch.float16,
    #         #         enabled=decode_parameter.device.type == "cuda",
    #         #     ):
    #         #         reconstruction = model.decoder(model.post_vq_conv(latent))
    #         #     del latent
    
    #         #     # 先移到 CPU，再转 float32，避免在 GPU 上额外创建一份 FP32 输出。
    #         #     reconstruction = reconstruction.cpu().float()
    #         #     print("reconstruction shape:", reconstruction.shape)
    #         #     volume = reconstruction[0, 0].permute(1, 2, 0).numpy()
    #         #     nib.save(nib.Nifti1Image(volume, affine), "reconstruction.nii.gz")
    
    #         # psnr = peak_signal_noise_ratio(reference.numpy()[0, 0], reconstruction.numpy()[0, 0])
    #         # # psnr_list.append(psnr)
    #         # print(f"PSNR: {psnr:.4f}")

    #     # break  # 只测试一个 batch，避免占用过多显存

    # # 