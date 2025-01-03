import torch
from torch import nn

from src.models.generator.AbstractGenerator import AbstractGenerator


class GeneratorLinearMappingSig(AbstractGenerator):

    def __init__(self, latent_size):
        super(GeneratorLinearMappingSig, self).__init__()

        self._noise_dim = torch.tensor([latent_size])

        self.image_height = 32

        self.network = nn.Sequential(
            nn.Linear(latent_size, 2*latent_size),
            nn.BatchNorm1d(2*latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(2*latent_size, 4*latent_size),
            nn.BatchNorm1d(4*latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(4*latent_size, 8*latent_size),
            nn.BatchNorm1d(8*latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(8*latent_size, 16*latent_size),
            nn.BatchNorm1d(16*latent_size),
            nn.Linear(16 * latent_size, self.image_height * self.image_height),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.network(x)
        return x

    def sample_subspace_masks(self, noise):
        return self.forward(noise).view(-1, 1, self.image_height, self.image_height).expand(-1, 3, -1, -1)
