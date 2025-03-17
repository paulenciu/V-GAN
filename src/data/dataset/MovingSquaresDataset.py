import torch

from src.data.IDataset import IDataset


class MovingSquaresDataset(IDataset):

    def __init__(self, root_dir, inlier_category=None, transform=None, train=True, image_size=(32, 32),
                 normalize=False):
        self.num_samples = 5000
        self.data = []
        self.labels = []
        self.image_size = image_size

        self._generate_inlier_data(image_size)

        if not train:
            # Generate outliers: circles or random noise
            num_outliers = 1000
            rnd_idx = torch.randint(0, self.num_samples, (num_outliers,))
            for idx in rnd_idx:
                image = torch.zeros(3, *image_size)
                # Draw a circle
                center_x = torch.randint(5, image_size[0] - 5, (1,))
                center_y = torch.randint(5, image_size[1] - 5, (1,))
                radius = 4
                for x in range(image_size[0]):
                    for y in range(image_size[1]):
                        if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                            image[:, x, y] = 1.0
                self.data[idx] = image
                self.labels[idx] = 1  # Outlier label

        self.image_shape = self.data[0].shape
        super(MovingSquaresDataset, self).__init__(image_shape=(3, *image_size), dataset=self.data, labels=self.labels)


    def _generate_inlier_data(self, image_size):
        height, width = image_size
        square_size = 4
        for _ in range(self.num_samples):
            image = torch.zeros(3, height, width)
            # Random top-left position
            x = torch.randint(0, height - square_size + 1, (1,))
            y = torch.randint(0, width - square_size + 1, (1,))
            image[:, x:x + square_size, y:y + square_size] = 1.0
            self.data.append(image)
            self.labels.append(0)  # Inlier label

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]