import torch
from torch import nn

from src.models.generator.IGenerator import IGenerator


class GeneratorLinearMappingTanh(IGenerator):

    def __init__(self, latent_size, image_height):
        super(GeneratorLinearMappingTanh, self).__init__()

        self.noise_dim = torch.tensor([latent_size])

        self.image_height = image_height

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
            nn.Tanh()
        )

    def forward(self, x):
        x = self.network(x)
        return x

    def get_noise_tensor_shape(self):
        return self.noise_dim

    def sample_subspace_masks(self, noise):
        return self.forward(noise).view(-1, 1, self.image_height, self.image_height).expand(-1, 3, -1, -1)
