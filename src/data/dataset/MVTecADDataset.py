import os
from typing import List

import torch
from torchvision import transforms
from PIL import Image

from src.data.IDataset import IDataset


class MVTecADDataset(IDataset):
    def __init__(self, file_path: str, inlier_category: str, train: bool = True, transform=None, normalize=False):
        """
        Initialize the MVTecADDataset.

        Args:
            file_path (str): Path to the dataset.
            category (List[str]): Categories of the dataset (e.g., ["bottle"]).
            train (bool): Whether to load the training set (default: True).
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.file_path = file_path
        self.category = inlier_category
        self.train = train

        if normalize:
            transform = transforms.Compose([
                transform,
               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])
            ])

        self.transform = transform

        # Initialize lists to store image paths and labels
        self.image_files = []
        self.labels = []

        # Load the dataset
        self._load_dataset()

        img, _ = self.__getitem__(0)
        img_shape = img.shape

        super().__init__(img_shape, self.image_files, self.labels)

    def _load_dataset(self):
        """
        Load the dataset and populate self.image_files and self.labels.
        """
        if self.train:
            image_dir = os.path.join(self.file_path, self.category, 'train', 'good')
            self.image_files.extend([
                os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')
            ])
            self.labels.extend([0] * len(os.listdir(image_dir)))  # 0 for normal images
        else:

            test_dir = os.path.join(self.file_path, self.category, 'test')
            for defect_type in os.listdir(test_dir):
                defect_dir = os.path.join(test_dir, defect_type)
                if os.path.isdir(defect_dir):
                    self.image_files.extend([
                        os.path.join(defect_dir, f) for f in os.listdir(defect_dir) if f.endswith('.png')
                    ])

                    self.labels.extend([0 if defect_type == 'good' else 1] * len(os.listdir(defect_dir)))

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get an image and its corresponding label by index.

        Args:
            idx (int): Index of the image.

        Returns:
            image (torch.Tensor): Transformed image.
            label (int): Label of the image (0 for normal, 1 for defective).
        """
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label