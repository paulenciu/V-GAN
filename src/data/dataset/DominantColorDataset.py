import random

import torch

from src.data.IDataset import IDataset


class DominantColorDataset(IDataset):

    def __init__(self, root_dir, inlier_category=None, transform=None, train=True, image_size=(32, 32),
                 normalize=False):
        if inlier_category is None:
            inlier_category = ["R", "G"]  # Default to Red and Green

        self.num_samples = 5000
        self.data = []
        self.labels = []
        self.category = inlier_category
        self.image_size = image_size

        self._generate_inlier_data(image_size)

        if not train:
            # Generate outliers: mixed colors or random noise
            num_outliers = 1000
            rnd_idx = torch.randint(0, self.num_samples, (num_outliers,))
            for idx in rnd_idx:
                # Create a mixed color (e.g., yellow: R+G)
                image = torch.zeros(3, *image_size)
                image[0] = 1.0  # Red
                image[1] = 1.0  # Green
                self.data[idx] = image
                self.labels[idx] = 1  # Outlier label

        self.image_shape = self.data[0].shape

    def _generate_inlier_data(self, image_size):
        height, width = image_size
        num_categories = len(self.category)
        samples_per_category = self.num_samples // num_categories

        for cat in self.category:
            for _ in range(samples_per_category):
                image = torch.zeros(3, height, width)
                if cat == "R":
                    image[0] = 1.0  # Red channel
                elif cat == "G":
                    image[1] = 1.0  # Green channel
                elif cat == "B":
                    image[2] = 1.0  # Blue channel
                self.data.append(image)
                self.labels.append(0)  # Inlier label

        # Handle remaining samples
        remaining = self.num_samples % num_categories
        for _ in range(remaining):
            cat = random.choice(self.category)
            # ... generate image as above ...
            self.data.append(image)
            self.labels.append(0)

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        super(DominantColorDataset, self).__init__(image_shape=(3, *image_size), dataset=self.data, labels=self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]