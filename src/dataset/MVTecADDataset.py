import os

import torch
import torchvision
from PIL import Image

from src.dataset.IDataset import IDataset


class MVTecADDataset(IDataset):

    def __init__(self, root_dir, category, transform=None, train=True):
        self.root_dir = root_dir
        self.category = category
        self.transform = transform
        self.train = train

        if self.train:
            self.image_dir = os.path.join(self.root_dir, self.category, 'train', 'good')
        else:
            self.image_dir = os.path.join(self.root_dir, self.category, 'test')

        self.image_files = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith('.png')]

        if self.transform is not None:
            image = self.transform(Image.open(self.image_files[0]))
        else:
            image = Image.open(self.image_files[0])

        if isinstance(image, torch.Tensor):
            image_shape = image.shape
        else:
            image_shape = torchvision.transforms.ToTensor()(image).shape

        super().__init__(image_shape, self.image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.category