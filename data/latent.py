"""Paired NCCT/CCTA latents exported by infer_vqgan.py."""
from pathlib import Path
import torch
from .base_dataset import BaseDataset
import torchio as tio
import numpy as np

class CT_Latent(BaseDataset):
    """Load one paired volume per item, with tensors in [C,D,H,W] order.

    Expected paths: dataset_dir/split/{ncct_latents,ccta_latents}/{ID}.pt.
    Files contain {'latent': Tensor, 'affine': Tensor or numpy.ndarray}.
    Only load trusted, locally exported files: NumPy affine metadata requires
    weights_only=False. No resize, normalization, flip or patch sampling occurs.
    BaseDataset supplies paired paths and __len__, as in CT3D.
    Its PIL transforms are unused because __getitem__ is overridden.
    resolution is metadata only; saved tensor dimensions are preserved.
    """

    def __init__(
        self, dataset_dir, split, resolution=256, seed=42,
        num_workers=0, 
    ):
        # Do not pass a 3D resolution to BaseDataset's unused 2D Resize.
        super().__init__(
            dataset_dir=str(Path(dataset_dir) / split),
            split=split,
            rgb_subfolder="ncct_latents",
            mask_subfolder="ccta_latents",
            rgb_name_mode=".pt",
            mask_name_mode=".pt",
            seed=seed,
        )
        self.resolution = resolution
        if not self.image_filenames:
            raise ValueError(f"Empty latent dataset: {self.dataset_dir}")
        missing = [path for path in self.mask_filenames if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing paired CCTA latents: {missing}")

        self.subjects = []
        for ncct, ccta, subject_id in zip(self.image_filenames, self.mask_filenames, self.base_filenames):
            self.subjects.append(tio.Subject(
                imgs=tio.ScalarImage(ncct, reader=self._read_latent),
                labels=tio.ScalarImage(ccta, reader=self._read_latent),
                ID=subject_id,
            ))

        self.dataset = tio.SubjectsDataset(self.subjects)

    @staticmethod
    def _read_latent(path):
        # Only trusted exports; affine metadata may contain a NumPy array.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        tensor = saved["latent"].float().permute(0, 2, 3, 1).contiguous()
        # CHWD latent index grid, not the decoded image's physical grid.
        return tensor, np.eye(4)
    
    def __getitem__(self, index):
        ncct = torch.load(self.image_filenames[index], map_location="cpu", weights_only=False)
        ccta = torch.load(self.mask_filenames[index], map_location="cpu", weights_only=False)
        # [C, D, H, W] -> [C, H, W, D]
        # 因为我们实现的unet，会先做permute
        imgs = ncct["latent"].permute(0, 2, 3, 1).float().contiguous()
        labels = ccta["latent"].permute(0, 2, 3, 1).float().contiguous()
        if imgs.ndim != 4 or imgs.shape != labels.shape:
            raise ValueError(
                f"{self.base_filenames[index]}: expected matching [C,D,H,W] latents, "
                f"got {tuple(imgs.shape)} and {tuple(labels.shape)}"
            )
        return {
            "imgs": imgs,
            "labels": labels,
            "ID": self.base_filenames[index],
            # These describe decoded image grids, not latent voxel spacing.
            "affine": torch.as_tensor(ncct["affine"], dtype=torch.float64),
            "target_affine": torch.as_tensor(ccta["affine"], dtype=torch.float64),
        }

    
if __name__ == '__main__':
    dataset = CT_Latent(dataset_dir='/data/jzk/CT/data_522', split="train", resolution=(128,128,32), seed=42, num_workers=2,)
    print(f"Dataset length: {len(dataset)}")

    for i in dataset:
        #x = i['imgs'].data
        # 3D
        # (1, 512, 512, 2)
        # torch.Size([1, 512, 512, 2]) <built-in method max of Tensor object at 0x7f79b24a5b70> <built-in method min of Tensor object at 0x7f79b24a5b70>

        # 2D
        # [2, 1, 512, 512]
        print(i.keys())
        # print(i['imgs'].shape)
        print(i["ID"], i['imgs'].data.shape, i['imgs'].data.max(), i['imgs'].data.min(), i['affine'].shape, i['target_affine'].shape, i["labels"].shape)
        break

    # for index in range(min(5, len(dataset.dataset))):
    #     original = dataset.subjects[index]
    #     transformed = dataset.dataset[index]

    #     print("ID:", original["ID"])
    #     for key in ("imgs", "labels"):
    #         path = original[key].path
    #         print(
    #             key,
    #             "\n  path:", path,
    #             "\n  nibabel shape:", nib.load(str(path)).shape,
    #             "\n  TorchIO before:", original[key].shape,
    #             "\n  TorchIO after:", transformed[key].shape,
    #         )