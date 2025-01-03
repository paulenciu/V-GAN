import torch
from torch import nn
import torch.nn.functional as F

from src.models.generator.AbstractGenerator import AbstractGenerator


class Generator2DRotationSmall(AbstractGenerator):

    def __init__(self, latent_size):
        super(Generator2DRotationSmall, self).__init__()

        if isinstance(latent_size, int):
            self._noise_dim = torch.tensor([latent_size])
        elif isinstance(latent_size, torch.Tensor):
            self._noise_dim = latent_size

        # Define the layers of the generator
        self.fc1 = nn.Linear(latent_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        # Pass the input through the layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # Normalize to ensure the vector lies on the unit circle
        rotation_matrix = self.f_function(x)
        return rotation_matrix

    def g_function(self, rotation_matrix):
        """
        Maps a 2D rotation matrix to the representation space.
        """
        return rotation_matrix[:, :, 0]  # Extract the first column (cos(theta), sin(theta))

    def f_function(self, vector):
        """
        Maps a 2D vector from the representation space back to a 2D rotation matrix.
        """
        vector = F.normalize(vector, dim=1)
        orthogonal_vector = torch.stack([-vector[:, 1], vector[:, 0]], dim=1)
        rotation_matrix = torch.stack([vector, orthogonal_vector], dim=2)
        return rotation_matrix

    def sample_subspace_masks(self, noise):
        return self.forward(noise)

