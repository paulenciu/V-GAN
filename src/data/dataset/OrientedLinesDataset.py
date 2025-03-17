import random

import torch

from src.data.IDataset import IDataset


class OrientedLinesDataset(IDataset):

    def __init__(self, root_dir, inlier_category=None, transform=None, train=True, image_size=(32, 32),
                 normalize=False):
        if inlier_category is None:
            inlier_category = ["0", "90"]  # Default to horizontal and vertical

        self.num_samples = 5000
        self.data = []
        self.labels = []
        self.category = inlier_category
        self.image_size = image_size

        self._generate_inlier_data(image_size)

        if not train:
            # Generate outliers: 45-degree lines or random noise
            num_outliers = 1000
            rnd_idx = torch.randint(0, self.num_samples, (num_outliers,))
            for idx in rnd_idx:
                if idx % 2 == 0:
                    # Create 45-degree line
                    image = torch.zeros(3, *image_size)
                    for i in range(image_size[0]):
                        if i < image_size[1]:
                            image[:, i, i] = 1.0
                    self.data[idx] = image
                else:
                    # Random noise
                    self.data[idx] = torch.rand(3, *image_size)
                self.labels[idx] = 1  # Outlier label

        self.image_shape = self.data[0].shape
        super(OrientedLinesDataset, self).__init__(image_shape=(3, *image_size), dataset=self.data, labels=self.labels)


    def _generate_inlier_data(self, image_size):
        height, width = image_size
        num_categories = len(self.category)
        samples_per_category = self.num_samples // num_categories

        for cat in self.category:
            for _ in range(samples_per_category):
                image = torch.zeros(3, height, width)
                if cat == "0":  # Horizontal line
                    row = height // 2
                    image[:, row, :] = 1.0
                elif cat == "90":  # Vertical line
                    col = width // 2
                    image[:, :, col] = 1.0
                elif cat == "45":  # Diagonal line (inlier if included)
                    for i in range(height):
                        if i < width:
                            image[:, i, i] = 1.0
                self.data.append(image)
                self.labels.append(0)  # Inlier label

        # Handle remaining samples
        remaining = self.num_samples % num_categories
        for _ in range(remaining):
            cat = random.choice(self.category)
            image = torch.zeros(3, height, width)
            # ... generate image as above ...
            self.data.append(image)
            self.labels.append(0)

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]