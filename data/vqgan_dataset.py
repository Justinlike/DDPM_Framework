import os
from os import path
from torch.utils.data import Dataset
import nibabel as nib
import torch
import torchio as tio
from torchio import transforms as T
from torchio.data import UniformSampler
from einops import rearrange

def data_transform_3d_m11():
    # define the [-1, 1] normalization transform and maybe we need a cut off range.
    def custom_standardize(image):
        # return [-1, 1] normalization
        # image.clamp(-1000, 1000)
        max =  image.max()
        min = image.min()
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


class VQGAN_Dataset(Dataset):
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
        ) -> None:
        super().__init__()
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

        self.dataset = tio.SubjectsDataset(self.subjects, transform=trans[split])

        self.queue = tio.Queue(
            subjects_dataset=self.dataset,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=UniformSampler(tuple(resolution)),
            num_workers=num_workers,
            shuffle_subjects=training,
            shuffle_patches=training,
        )

    def __len__(self):
        return len(self.queue)

    def __getitem__(self, index):
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
        print(i["data"].shape, i['imgs'].data.max, i['imgs'].data.min)  # Should print (C, T, H, W)