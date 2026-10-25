from torch.nn.modules import ConvTranspose2d
# from .brain import Brain
from .ct import CT2D, CT3D
from .base_dataset import BaseDataset
from torch.utils.data import DataLoader
import torch
import numpy as np

dataset_type_dict: dict[str, type[BaseDataset]] = {
    "ct": CT2D,
    "ct3d": CT3D
}

def build_loader(cfg):
    train_dataset, vaild_dataset, test_dataset = build_dataset(
        base_data_dir=cfg.base_data_dir, 
        dataset_type=cfg.dataset_name,
        resolution=cfg.resolution,
        seed=cfg.seed,
    )

    train_dataloader = DataLoader(
        train_dataset,
        cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    valid_dataloader = DataLoader(
        vaild_dataset,
        cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=True
    )
    return train_dataset, train_dataloader, vaild_dataset, valid_dataloader, test_dataset, test_dataloader

def collate_fn(examples):
    pixel_values = torch.stack([_to_tensor(example["imgs"]) for example in examples], dim=0)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    mask_values = torch.stack([_to_tensor(example["labels"]) for example in examples], dim=0)
    mask_values = mask_values.to(memory_format=torch.contiguous_format).float()

    return {"imgs": pixel_values, "labels": mask_values}

def _to_tensor(x):
    # TorchIO 的 ScalarImage / LabelMap 转底层 torch.Tensor，其他类型原样返回
    if hasattr(x, "data") and not isinstance(x, torch.Tensor):
        return x.data
    return x

def build_dataset(
        base_data_dir: str,
        dataset_type: str,
        resolution: int = 256,
        seed: int = 42,
        **kwargs
):
    data_type = dataset_type_dict[dataset_type]
    dataset_dir = base_data_dir
    return (
        data_type(dataset_dir=dataset_dir, split="train", resolution=resolution, seed=seed, **kwargs),
        data_type(dataset_dir=dataset_dir, split="test", resolution=resolution, seed=seed, **kwargs),
        data_type(dataset_dir=dataset_dir, split="test", resolution=resolution, seed=seed, **kwargs)
    )