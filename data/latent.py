import os
from .base_dataset import BaseDataset
import nibabel as nib
import torch
import torchio as tio
from torchio import transforms as T
from torchio.data import UniformSampler
from einops import rearrange

def data_transform_3d():
    # define the min-max normalization transform
    # [-1, 1] transformation achieved in Baser class.
    def custom_standardize(image):
        max = image.max()
        min = image.min()
        return (image - min) / (max - min)

    def resize_hw(image):
        B, H, W, D = image.shape
        return T.Resize((256, 256, D), image_interpolation='linear')(image)
    
    data_transform = {
        "train": T.Compose([
            # # tio 中，resize的-1表示维度不变
            # tio.Lambda(resize_hw),
            # T.Resize((256, 256, -1), image_interpolation='linear'),
            T.RandomFlip(),
            # tio.Lambda(custom_standardize),
        ]),
        "val": T.Compose([
            # tio.Lambda(resize_hw),
            # T.Resize((256, 256, -1), image_interpolation='linear'),
            # tio.Lambda(custom_standardize),
        ]),
        "test": T.Compose([
            # tio.Lambda(resize_hw),
            # T.Resize((256, 256, -1), image_interpolation='linear'),
            # tio.Lambda(custom_standardize),
        ]),
    }

    return data_transform

class CT_Latent(BaseDataset):
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
        # 继承BaseDataset只起到加载数据集的作用，主要是获取ncct和ccta的路径，不对数据处理
        # 
        super().__init__(
            rgb_subfolder="ncct_latents",
            mask_subfolder="ccta_latents",
            rgb_name_mode=".pt",
            mask_name_mode=".pt",
            **kwargs, 
        )
        resolution_3d = kwargs["resolution"]

        self.subjects = []

        print(len(self.image_filenames))
        print(len(self.mask_filenames))
        
        for ncct, ccta, file in zip(self.image_filenames, self.mask_filenames, self.base_filenames):
            # /data/jzk/CT/data_cba/train/ncct/1079294.nii.gz /data/jzk/CT/data_cba/train/ccta/1079294.nii.gz 1079294
            # 也许再加一个affine矩阵
            subject = tio.Subject(
                imgs=torch.load(
                    ncct, 
                    map_location="cpu",
                    weights_only=True,
                )["latent"].float(), 
                labels=torch.load(
                    ccta, 
                    map_location="cpu", 
                    weights_only=True
                )["latent"].float(),
                ID=file
            )
            self.subjects.append(subject)
        
        trans = data_transform_3d()

        # load the augment and norm policy
        self.dataset = tio.SubjectsDataset(self.subjects, transform=trans[split])

        # 从整幅图像中抽取 patch，缓存起来, patch size 就是传入的resolution
        self.queue_set = tio.Queue(
            subjects_dataset=self.dataset,
            max_length=queue_length,                # 队列中最多缓存的patch数
            samples_per_volume=samples_per_volume,  # 抽取patch的数量
            sampler=UniformSampler(resolution_3d),  # 抽取patch的分辨率，以供最后拼回去
            num_workers=num_workers,
            # shuffle_subjects=shuffle_subjects,  # 是否打乱subject的顺序，训练时为True，测试时为False
            # shuffle_patches=shuffle_patches,    # 是否打乱patch的顺序，
            shuffle_subjects=True if split == "train" else False,  # 是否打乱subject的顺序，训练时为True，测试时为False
            shuffle_patches=True if split == "train" else False,   # 是否打乱patch的顺序，训练时为True，测试时为False
        )
    
        
    def __getitem__(self, index):
        # 不知道是否需要返回affine
        return self.queue_set[index]
    
if __name__ == '__main__':
    dataset = CT_Latent(dataset_dir='/data/jzk/CT/data_522', split="test", resolution=(128,128,32), seed=42, num_workers=2, queue_length=16, samples_per_volume=16)
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
        print(i["ID"], i['imgs'].data.shape, i['imgs'].data.max(), i['imgs'].data.min())
        # break

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