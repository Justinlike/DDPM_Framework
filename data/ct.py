import os
from .base_dataset import BaseDataset
import nibabel as nib
import torch
import torchio as tio
from torchio import transforms as T
from torchio.data import UniformSampler
from einops import rearrange

def data_transform_3d():
    normalization = tio.ZNormalization()

    # define the min-max normalization transform
    def custom_standardize(image):
        max = image.max()
        min = image.min()
        return (image - min) / (max - min)

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

class CT2D(BaseSegDataset):
    def __init__(
        self,
        queue_length = 16,
        samples_per_volume = 16,
        patch_size = (96, 96, 64),
        num_workers = 8,
        shuffle_subjects = True,
        shuffle_patches = True,
        **kwargs
    ):
        super().__init__(
            dataset_dir=kwargs.get("dataset_dir"),
            split=kwargs.get("split"),
            rgb_subfolder="ncct",
            mask_subfolder="ccta",
            rgb_name_mode=".nii.gz",
            mask_name_mode=".nii.gz",
            **kwargs
        )
        dateset_dir = kwargs.get("dataset_dir")
        split = kwargs.get("split")
        kwargs["dataset_dir"] = os.path.join(dateset_dir, split) # data/split 文件夹
        resolution_3d = kwargs['resolution']
        
        pass
    
    def __getitem__(self, index):
        subject = self.dataset[index]
        image = subject.imgs.data
        mask = subject.labels.data

        # Rearrange the dimensions to (H, W, D)
        image = rearrange(image, 'c h w d -> h w d c')
        mask = rearrange(mask, 'c h w d -> h w d c')

        return image, mask


class CT3D(BaseSegDataset):

    def __init__(
        self,
        queue_length=16,
        samples_per_volume=5,
        patch_size=(96,96,64),
        num_workers=8,
        shuffle_subjects=True,
        shuffle_patches=True,
        **kwargs,
    ) -> None:
        dataset_dir = kwargs.get("dataset_dir")
        split = kwargs.get("split")
        kwargs["dataset_dir"] = os.path.join(dataset_dir, split)
        super().__init__(
            rgb_subfolder="ncct",
            mask_subfolder="ccta",
            rgb_name_mode=".nii.gz",
            mask_name_mode=".nii.gz",
            **kwargs
        )
        resolution_3d = kwargs['resolution']
       # resolution_3d = (96,96,64)
                # Preprocessing the datasets

        self.subjects = []

        for ncct, ccta, file in zip(self.image_filenames, self.mask_filenames, self.base_filenames):
            subject = tio.Subject(imgs=tio.ScalarImage(ncct), labels=tio.ScalarImage(ccta), ID=file)
            self.subjects.append(subject)

        trans = data_transform_3d()
        self.dataset = tio.SubjectsDataset(self.subjects, transform=trans[split])

        self.queue_set = tio.Queue(
            subjects_dataset=self.dataset,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=UniformSampler(resolution_3d),
           # num_workers=num_workers,
            shuffle_subjects=shuffle_subjects,
            shuffle_patches=shuffle_patches
        )

        # self.image_transforms = image_transforms
        # self.mask_transforms = mask_transforms
    
    def __getitem__(self, index):
        return self.queue_set[index]
