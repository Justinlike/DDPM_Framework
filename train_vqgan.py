"Adapted from https://github.com/SongweiGe/TATS"

import os
from typing import Any
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
import torch
from models.vqgan_3d import VQGAN
# from train.callbacks import ImageLogger, VideoLogger
from data.build import build_loader
import hydra
from data.build import dataset_type_dict
from omegaconf import DictConfig, open_dict
# import wandb

# ========================== 先对CT3D类包装， 使其输出符合MedicalDiffusion的batch格式 ==========================
class VQGANDatasetAdapter(Dataset):
    """Convert this project's TorchIO samples to MedicalDiffusion batches."""

    def __init__(self, dataset: Dataset, image_key: str, normalization: str):
        # CT3D stores the actual patch dataset in a TorchIO Queue. Using the
        # queue directly also preserves samples_per_volume in __len__.
        self.dataset = getattr(dataset, "queue_set", dataset)
        self.image_key = image_key
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        image: Any = sample[self.image_key]
        image = image.data if hasattr(image, "data") else image
        image = torch.as_tensor(image).float()

        if image.ndim != 4:
            raise ValueError(
                f"Expected a (C, H, W, D) CT patch, got shape {tuple(image.shape)}"
            )

        # TorchIO: (C, H, W, D); MedicalDiffusion: (C, T, H, W).
        image = image.permute(0, 3, 1, 2).contiguous()

        if self.normalization == "zero_one_to_minus_one":
            image = image.mul(2.0).sub(1.0)
        elif self.normalization != "none":
            raise ValueError(f"Unknown dataset normalization: {self.normalization}")

        return {"data": image}


def _make_dataset(cfg: DictConfig, split: str, training: bool) -> Dataset:
    try:
        dataset_class = dataset_type_dict[cfg.dataset.type]
    except KeyError as exc:
        available = ", ".join(sorted(dataset_type_dict))
        raise ValueError(
            f"Unknown dataset type '{cfg.dataset.type}'. Available: {available}"
        ) from exc

    dataset = dataset_class(
        dataset_dir=cfg.dataset.root_dir,
        split=split,
        resolution=tuple(cfg.dataset.resolution),
        seed=cfg.model.seed,
        queue_length=cfg.dataset.queue_length,
        samples_per_volume=cfg.dataset.samples_per_volume,
        shuffle_subjects=training,
        shuffle_patches=training,
    )
    return VQGANDatasetAdapter(
        dataset,
        image_key=cfg.dataset.image_key,
        normalization=cfg.dataset.normalization,
    )


def _make_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    train_dataset = _make_dataset(cfg, cfg.dataset.train_split, training=True)
    val_dataset = _make_dataset(cfg, cfg.dataset.val_split, training=False)
    loader_kwargs = {
        "batch_size": cfg.model.batch_size,
        "num_workers": cfg.model.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.model.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def _latest_checkpoint(root_dir: str) -> str | None:
    """Return the newest Lightning checkpoint without renaming user files."""
    checkpoint_root = Path(root_dir) / "lightning_logs"
    if not checkpoint_root.exists():
        return None
    checkpoints = list(checkpoint_root.glob("version_*/checkpoints/*.ckpt"))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda path: path.stat().st_mtime))


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)

    # 接着写
    train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader = build_loader(cfg)

    # automatically adjust learning rate
    batch_size, base_lr, ngpu, accumulate = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus, cfg.model.accumulate_grad_batches

    with open_dict(cfg):
        cfg.model.lr = accumulate * (ngpu/8.) * (batch_size/4.) * base_lr
        cfg.model.default_root_dir = os.path.join(
            cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix)
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
        cfg.model.lr, accumulate, ngpu/8, batch_size/4, base_lr))

    model = VQGAN(cfg)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
                     filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    # callbacks.append(ImageLogger(
    #     batch_frequency=750, max_images=4, clamp=True))
    # callbacks.append(VideoLogger(
    #     batch_frequency=1500, max_videos=4, clamp=True))

    # load the most recent checkpoint file
    base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir):
        log_folder = ckpt_file = ''
        version_id_used = step_used = 0
        for folder in os.listdir(base_dir):
            version_id = int(folder.split('_')[1])
            if version_id > version_id_used:
                version_id_used = version_id
                log_folder = folder
        if len(log_folder) > 0:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            for fn in os.listdir(ckpt_folder):
                if fn == 'latest_checkpoint.ckpt':
                    ckpt_file = 'latest_checkpoint_prev.ckpt'
                    os.rename(os.path.join(ckpt_folder, fn),
                              os.path.join(ckpt_folder, ckpt_file))
            if len(ckpt_file) > 0:
                cfg.model.resume_from_checkpoint = os.path.join(
                    ckpt_folder, ckpt_file)
                print('will start from the recent ckpt %s' %
                      cfg.model.resume_from_checkpoint)

    accelerator = None
    if cfg.model.gpus > 1:
        accelerator = 'ddp'

    trainer = pl.Trainer(
        gpus=cfg.model.gpus,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        resume_from_checkpoint=cfg.model.resume_from_checkpoint,
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        gradient_clip_val=cfg.model.gradient_clip_val,
        accelerator=accelerator,
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    run()

    # PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=brats dataset.root_dir=<INSERT_PATH_TO_BRATS_DATASET> model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='flair' model.precision=16 model.embedding_dim=8 model.n_hiddens=16 model.downsample=[2,2,2] model.num_workers=32 model.gradient_clip_val=1.0 model.lr=3e-4 model.discriminator_iter_start=10000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384 model.accumulate_grad_batches=1 