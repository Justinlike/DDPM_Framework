import os
from base_dataset import BaseDataset
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

class CT2D(BaseDataset):
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
        dataset_dir = kwargs.get("dataset_dir")
        split = kwargs.get("split")
        kwargs["dataset_dir"] = os.path.join(dataset_dir, split)
        super().__init__(
            rgb_subfolder="ncct",
            mask_subfolder="ccta",
            rgb_name_mode=".nii.gz",
            mask_name_mode=".nii.gz",
            **kwargs, 
        )
        resolution_3d = kwargs["resolution"]

        self.subjects = []

        for ncct, ccta, file in zip(self.image_filenames, self.mask_filenames, self.base_filenames):
            # /data/jzk/CT/data_cba/train/ncct/1079294.nii.gz /data/jzk/CT/data_cba/train/ccta/1079294.nii.gz 1079294
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

    
    def __getitem__(self, index):
        sample = self.queue_set[index]

        ncct = sample["imgs"].data
        ccta = sample["labels"].data

        ncct = rearrange(ncct, 'c x y z -> z c x y')
        ccta = rearrange(ncct, 'c x y z -> z c x y')

        return {"imgs":ncct, "labels":ccta}


class CT3D(BaseDataset):
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
        dataset_dir = kwargs.get("dataset_dir")
        split = kwargs.get("split")
        kwargs["dataset_dir"] = os.path.join(dataset_dir, split)
        super().__init__(
            rgb_subfolder="ncct",
            mask_subfolder="ccta",
            rgb_name_mode=".nii.gz",
            mask_name_mode=".nii.gz",
            **kwargs, 
        )
        resolution_3d = kwargs["resolution"]

        self.subjects = []

        for ncct, ccta, file in zip(self.image_filenames, self.mask_filenames, self.base_filenames):
            # /data/jzk/CT/data_cba/train/ncct/1079294.nii.gz /data/jzk/CT/data_cba/train/ccta/1079294.nii.gz 1079294
            subject = tio.Subject(imgs=tio.ScalarImage(ncct), labels=tio.ScalarImage(ccta), ID=file)
            self.subjects.append(subject)
        trans = data_transform_3d()

        # load the augment and norm policy
        self.dataset = tio.SubjectsDataset(self.subjects, transform=trans[split])

        # 从整幅图像中抽取 patch，缓存起来, patch size 就是传入的resolution
        self.queue_set = tio.Queue(
            subjects_dataset=self.dataset,
            max_length=queue_length,                # 队列中最多缓存的patch数
            samples_per_volume=samples_per_volume,  # 抽取patch的数量
            sampler=UniformSampler(resolution_3d),  # 抽取patch的分辨率
            # num_workers=num_workers,
            shuffle_subjects=shuffle_subjects,
            shuffle_patches=shuffle_patches
        )
    
        
    def __getitem__(self, index):
        return self.queue_set[index]

if __name__ == '__main__':
    dataset = CT2D(dataset_dir='/data/jzk/CT/data_cba', split="train", resolution=(512,512,2), seed=42)

    for i in dataset:
        #x = i['imgs'].data
        # print(i['imgs'].shape)
        # 3D
        # (1, 512, 512, 2)
        # torch.Size([1, 512, 512, 2]) <built-in method max of Tensor object at 0x7f79b24a5b70> <built-in method min of Tensor object at 0x7f79b24a5b70>

        # 2D
        # [2, 1, 512, 512]
        print(i['imgs'].data.shape, i['imgs'].data.max, i['imgs'].data.min)