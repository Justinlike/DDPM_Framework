import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
form torchvision import transforms

class BaseDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split: str,
        rgb_subfolder: str,
        mask_subfolder: str,
        resolution: int = 256,
        rgb_name_mode: str = ".jpg",
        mask_name_mode: str = ".png",
        train_index: int = None,
        valid_index: int = None,
        seed: int = 42
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        assert os.path.exists(self.dataset_dir), f"Dataset does not exist at: {self.dataset_dir} "
        self.split = split

        image_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ])

        mask_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ])

        self.iamge_trasform = image_transform
        self.mask_transform = mask_transform

        self.rgb_subfolder = rgb_subfolder
        self.mask_subfolder = mask_subfolder
        self.resolution = resolution
        self.rgb_name_mode = rgb_name_mode
        self.mask_name_mode = mask_name_mode

        # load base filenames
        # 先拼接出目标文件夹，然后将目标后缀的文件取出来
        # basename就是
        base_filenames = os.listdir(os.path.join(self.dataset_dir, self.rgb_subfolder)) 
        self.base_filenames = [os.path.basename(f).split('.')[0] for f in base_filenames if f.endswith(self.rgb_name_mode)]

        self.train_index = train_index
        self.valid_index = valid_index
        self.seed = seed

        # split dataset into train, valid, and test sets
        if self.train_index is None:
            assert split in["train", "val", "test"], f"Invalid split mode: {split}"
            self.base_filenames = self._split(self.base_filenames, self.base_filenames)

        # load rgb and mask paths
        self.image_filenames = [os.path.join(dataset_dir, self.rgb_subfolder, f"{base_filename}{self.rgb_name_mode}") for base_filename in self.base_filenames]
        self.mask_filenames = [os.path.join(dataset_dir, self.mask_subfolder, f"{base_filename}{self.mask_name_mode}") for base_filename in self.base_filenames]

    def _split(self, image_paths, mask_paths):
        if self.split == 'train':
            return image_paths[:self.train_index], mask_paths[:self.train_index]
        elif self.split == 'valid':
            return image_paths[self.train_index:self.valid_index], mask_paths[self.train_index:self.valid_index]
        else:
            return image_paths[self.valid_index:], mask_paths[self.valid_index:]
    
    def _flip(self, rgb: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:
            rgb, mask = F.hflip(rgb), F.hflip(mask)
        if random.random() > 0.5:
            rgb, mask = F.vflip(rgb), F.vflip(mask)
        return rgb, mask
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_filenames[idx]).convert("RGB")
        mask = Image.open(self.mask_filenames[idx]).convert("RGB")
        image = self.iamge_trasform(image)
        mask = self.mask_transform(mask)

        # 只对训练数据集增强
        if self.split == "train":
            image, mask = self._flip(image, mask)

        return {
            "img": image,
            "mask": mask
        }