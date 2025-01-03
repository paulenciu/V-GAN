import torch
from torch import nn

from src.models.Generator import upper_softmax
from src.models.generator.AbstractGenerator import AbstractGenerator


class GeneratorOneChannel(AbstractGenerator):

    def __init__(self, latent_size):
        super(GeneratorOneChannel, self).__init__()

        self._noise_dim = torch.tensor([latent_size])

        self.layers = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size),
            nn.BatchNorm1d(2 * latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * latent_size, 4 * latent_size),
            nn.BatchNorm1d(4 * latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(4 * latent_size, 8 * latent_size),
            nn.BatchNorm1d(8 * latent_size),
            nn.Linear(8 * latent_size, 8 * latent_size),
            nn.BatchNorm1d(8 * latent_size),
            nn.Linear(8 * latent_size, 32*32),
            upper_softmax()
        )

    def forward(self, input):
        x = self.layers(input)
        return x

    def sample_subspace_masks(self, noise):
        return self.forward(noise).repeat(1, 3).view(-1, 3, 32, 32)