"Adapted from https://github.com/SongweiGe/TATS"

import os
from typing import Any
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch
from models.vqgan_3d import VQGAN
# from train.callbacks import ImageLogger, VideoLogger
from data.build import build_loader
import hydra
from omegaconf import DictConfig, open_dict
from data.vqgan_dataset import VQGAN_Dataset
# import wandb

def _make_dataset(cfg: DictConfig, split: str, ) -> Dataset:
    # root_dit = cfg.dataset.root_dir
    # train_split = cfg.dataset.train_split
    # val_split =  cfg.dataset.test_split

    train_dataset_0 = VQGAN_Dataset(
        root_dir="/data/jzk/CT/data_522",
        split="train",
        modality="ncct",
        resolution=(512, 512, 64),
        queue_length=16,
        num_workers=0,
        training=True,
        normalization="m11"
    )

    train_dataset_1 = VQGAN_Dataset(
        root_dir="/data/jzk/CT/data_522",
        split="train",
        modality="ccta",
        resolution=(512, 512, 64),
        queue_length=16,
        num_workers=0,
        training=True,
        normalization="m11"
    )

    val_dataset_0 = VQGAN_Dataset(
        root_dir="/data/jzk/CT/data_522",
        split="test",
        modality="ccta",
        resolution=(512, 512, 64),
        queue_length=16,
        num_workers=0,
        training=True,
        normalization="m11"
    )

    val_dataset_1 = VQGAN_Dataset(
        root_dir="/data/jzk/CT/data_522",
        split="test",
        modality="ccta",
        resolution=(512, 512, 64),
        queue_length=16,
        num_workers=0,
        training=True,
        normalization="m11"
    )

    return train_dataset_0, train_dataset_1, val_dataset_0, val_dataset_1

def _make_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    train_ncct, train_ccta, val_ncct, val_ccta = _make_dataset(
        cfg, cfg.dataset.train_split
    )

    # 合并训练集
    train_dataset = ConcatDataset([
        train_ncct,
        train_ccta
    ])

    # 合并验证集
    val_dataset = ConcatDataset([
        val_ncct,
        val_ccta
    ])

    loader_kwargs = {
        "batch_size": cfg.model.batch_size,
        "num_workers": cfg.model.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.model.num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )

    print(f"Train NCCT: {len(train_ncct)}")
    print(f"Train CCTA: {len(train_ccta)}")
    print(f"Train Total: {len(train_dataset)}")

    print(f"Val NCCT: {len(val_ncct)}")
    print(f"Val CCTA: {len(val_ccta)}")
    print(f"Val Total: {len(val_dataset)}")

    return train_loader, val_loader

@hydra.main(config_path='/data/jzk/DDPM_Framework/conf', config_name='config-vqgan', version_base=None)
def run(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)

    # 接着写
    train_dataloader, val_dataloader = _make_dataloaders(cfg)

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

    # GPU 配置
    num_gpus = cfg.model.gpus
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = num_gpus if accelerator == "gpu" else 1
    # 多 GPU 时使用 DDP
    strategy = "ddp" if devices > 1 else "auto"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,

        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        callbacks=callbacks,

        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        gradient_clip_val=cfg.model.gradient_clip_val,
        
    )
        
    trainer.fit(model, train_dataloader, val_dataloader, )


if __name__ == '__main__':
    run()

    # PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=brats dataset.root_dir=<INSERT_PATH_TO_BRATS_DATASET> model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='flair' model.precision=16 model.embedding_dim=8 model.n_hiddens=16 model.downsample=[2,2,2] model.num_workers=32 model.gradient_clip_val=1.0 model.lr=3e-4 model.discriminator_iter_start=10000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384 model.accumulate_grad_batches=1 