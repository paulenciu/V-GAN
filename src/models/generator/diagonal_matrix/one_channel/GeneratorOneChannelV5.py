import torch
from torch import nn

from src.models.Generator import upper_softmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator


class GeneratorOneChannelV5(AbstractGenerator):

    def __init__(self, latent_size):
        super(GeneratorOneChannelV5, self).__init__()

        self._noise_dim = torch.tensor([latent_size])

        self.layers = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2 * latent_size),

            nn.Linear(2 * latent_size, 4 * latent_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(4 * latent_size),

            nn.Linear(4 * latent_size, 8 * latent_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(8 * latent_size),

            nn.Linear(8 * latent_size, 32 * 32),
            nn.Sigmoid()
        )

    def forward(self, input, mode="train"):
        if mode == "train":
            return self.layers(input)
        return self.layers(input)

    def sample_subspace_masks(self, noise, mode="train"):
        return self.forward(noise, mode).repeat(1, 3).view(-1, 3, 32, 32)