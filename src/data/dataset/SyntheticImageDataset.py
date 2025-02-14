import torch
from src.data.IDataset import IDataset
from torch.utils.data import Dataset

class SyntheticImageDataset(Dataset):

    def __init__(self, root_dir, inlier_category=None, transform=None, train=True, image_size=(64, 64), normalize=False):
        """
        Initialize the data.

        Args:
            num_samples (int): Total number of samples in the data.
        """
        super(SyntheticImageDataset, self).__init__()
        if inlier_category is None:
            inlier_category = ["1", "2"]

        self.num_samples = 500
        self.data = []
        self.labels = []
        self.category = inlier_category
        self._generate_data(image_size)

        self.image_shape = self.data[0].shape

    def _generate_data(self, image_size):
        """
        Generate the synthetic data.
        """
        height, width = image_size
        half_height = int(height / 2)
        for _ in range(self.num_samples // 2):

            if "1" in self.category:
                # Upper half white, lower half black
                image = torch.zeros(3, height, width)
                image[:, :half_height, :] = 1.0  # Set upper half to white
                self.data.append(image)
                self.labels.append(0)  # Label: 0

            if "2" in self.category:
                # Upper half black, lower half white
                image = torch.zeros(3, height, width)
                image[:, half_height:, :] = 1.0  # Set lower half to white
                self.data.append(image)
                self.labels.append(1)  # Label: 1

        # Convert lists to tensors
        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        """
        Return the total number of samples.

        Returns:
            int: Total number of samples in the data.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a sample from the data.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, label) where image is a tensor of shape (3, 32, 32),
                   and label is the corresponding label.
        """
        return self.data[idx], self.labels[idx]
