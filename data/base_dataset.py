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
        base_filenames = os.listdir(os.path.join(self.dataset_dir, self.rgb_subfolder))
        self.base_filenames = [os.path.basename(f).split('.')[0] for f in base_filenames if f.endswith(self.rgb_name_mode)]

        self.train_index = train_index
        self.valid_index = valid_index
        self.seed = seed

        # load rgb and mask paths
        self.image_filenames = [os.path.join(dataset_dir, self.rgb_subfolder, f"{base_filename}{self.rgb_name_mode}") for base_filename in self.base_filenames]
        self.mask_filenames = [os.path.join(dataset_dir, self.mask_subfolder, f"{base_filename}{self.mask_name_mode}") for base_filename in self.base_filenames]
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image