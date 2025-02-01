import torch
from src.data.IDataset import IDataset
from torch.utils.data import Dataset

class SyntheticImageDataset(Dataset):

    def __init__(self, num_samples, lower_half_white=True, upper_half_white=True, image_size=(224, 224)):
        """
        Initialize the data.

        Args:
            num_samples (int): Total number of samples in the data.
        """
        super(SyntheticImageDataset, self).__init__()
        self.lower_half_white = lower_half_white
        self.upper_half_white = upper_half_white
        self.num_samples = num_samples
        self.data = []
        self.labels = []
        self._generate_data(image_size)

        self.image_shape = self.data[0].shape

    def _generate_data(self, image_size):
        """
        Generate the synthetic data.
        """
        height, width = image_size
        half_height = int(height / 2)
        for _ in range(self.num_samples // 2):

            if self.upper_half_white:
                # Upper half white, lower half black
                image = torch.zeros(3, height, width)
                image[:, :half_height, :] = 1.0  # Set upper half to white
                self.data.append(image)
                self.labels.append(0)  # Label: 0

            if self.lower_half_white:
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
