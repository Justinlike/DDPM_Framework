import os
from numbers import Integral
import torch
import torchio as tio
from torchio import transforms as T
from torchio.data import UniformSampler
from data.base_dataset import BaseDataset

class PadToMinimumSize(tio.Transform):
    """Pad undersized axes without cropping larger volumes before sampling."""

    def __init__(self, minimum_size):
        super().__init__()
        self.minimum_size = tuple(minimum_size)

    def apply_transform(self, subject):
        padding = []
        for size, minimum in zip(subject.spatial_shape, self.minimum_size):
            missing = max(0, minimum - size)
            padding.extend((missing // 2, missing - missing // 2))
        if any(padding):
            # Applied after normalization: -1 is the background intensity.
            subject = tio.Pad(tuple(padding), padding_mode=-1)(subject)
        return subject

def data_transform_3d_m11():
    # define the [-1, 1] normalization transform and maybe we need a cut off range.
    def custom_standardize(image):
        # return [-1, 1] normalization
        # image.clamp(-1000, 1000)
        max =  image.max()
        min = image.min()
        if max == min:
            return torch.full_like(image, -1)
        return (image - min) / (max - min) * 2.0 - 1.0

    data_transform = {
        "train": T.Compose([
            T.RandomFlip(),
            tio.Lambda(custom_standardize),
        ]),
        "val": T.Compose([
            tio.Lambda(custom_standardize),
        ]),
        "test": T.Compose([
            tio.Lambda(custom_standardize),
        ]),
    }

    return data_transform

class VQGAN_Dataset(BaseDataset):
    """Sample 3D patches: resolution is (H, W, D), output is (C, D, H, W).

    The inherited PIL transforms are unused. Volumes keep their voxel spacing;
    undersized axes are padded, and larger axes are randomly sampled.
    """
    def __init__(
            self, 
            root_dir: str,
            split: str,
            modality: str = "ncct",
            resolution: tuple[int, int, int] = (128, 128, 128),
            samples_per_volume: int = 16,
            queue_length: int = 16,
            num_workers: int = 8,
            training: bool = False,
            normalization: str = "m11",
            **kwargs
        ) -> None:
        if (not isinstance(resolution, (tuple, list)) or len(resolution) != 3
                or any(isinstance(size, bool) or not isinstance(size, Integral)
                       or size <= 0 for size in resolution)):
            raise ValueError("resolution must contain three positive integers (H, W, D)")
        if normalization != "m11":
            raise ValueError("VQGAN_Dataset currently supports only normalization='m11'")
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split: {split}")
        kwargs["split"] = split
        kwargs["dataset_dir"] = os.path.join(root_dir, split)
        super().__init__(
            rgb_subfolder=modality,
            mask_subfolder="ccta",
            rgb_name_mode=".nii.gz",
            mask_name_mode=".nii.gz",
            **kwargs, 
        )
        # Do not pass the 3D patch size to BaseDataset's 2D Resize.
        self.patch_size = tuple(int(size) for size in resolution)
        self.resolution = self.patch_size
        self.data_dir = os.path.join(root_dir, split, modality)

        # 发现有预处理的问题，不是所有的图像都是大于 [512, 512, 64]的
        # check if the data directory exists
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"VQGAN data directory does not exist: {self.data_dir}")

        # get all basenames of the files in the data directory
        # ['125410.nii.gz', '284758.nii.gz',]
        base_filenames = [
            file_path for file_path in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, file_path)) and file_path.endswith(".nii.gz")
        ]
        # print(len(base_filenames))

        self.base_filenames = [os.path.basename(f).split('.')[0] for f in base_filenames if f.endswith("nii.gz")]
        # print(self.base_filenames)

        if not base_filenames:
            raise RuntimeError(f"No NIfTI files found in: {self.data_dir}")

        self.subjects = []
        # print(f"Loading {len(base_filenames)} subjects from {self.data_dir}")
        for data, file in zip(base_filenames, self.base_filenames):
            # print(f"Loading subject: {data}, {file} from {self.data_dir}")
            subject = tio.Subject(data = tio.ScalarImage(os.path.join(self.data_dir, data)), ID = file)
            self.subjects.append(subject)
        print(f"Loaded {len(self.subjects)} subjects from {self.data_dir}")
        trans = data_transform_3d_m11()

        self.dataset = tio.SubjectsDataset(
            self.subjects,
            transform=T.Compose([trans[split], PadToMinimumSize(self.patch_size)]),
        )

        self.queue = tio.Queue(
            subjects_dataset=self.dataset,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=UniformSampler(self.patch_size),
            num_workers=num_workers,
            shuffle_subjects=training,
            shuffle_patches=training,
        )

    def __len__(self):
        return len(self.queue)

    def __getitem__(self, index):
        # Queue ignores indices, so enforce the Dataset iteration boundary here.
        if index < 0 or index >= len(self):
            raise IndexError(index)
        image = self.queue[index]["data"].data.float()
        # TorchIO uses (C, H, W, D); VQGAN expects (C, T, H, W).
        image = image.permute(0, 3, 1, 2).contiguous()
        return {"data": image}

if __name__ == "__main__":
    dataset = VQGAN_Dataset(
        root_dir="/data/jzk/CT/data_522",
        split="train",
        modality="ncct",
        resolution=(512, 512, 64),
        queue_length=16,
        num_workers=0,
        training=True,
        normalization="m11"
    )
    print(f"Dataset length: {len(dataset)}")
    for i in dataset:
        print(i["data"].shape, i['data'].data.max(), i['data'].data.min())  # Should print (C, T, H, W)
