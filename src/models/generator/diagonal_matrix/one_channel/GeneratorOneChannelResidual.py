import torch
from src.models.generator.AbstractGenerator import AbstractGenerator
from torch import nn

from src.models.Generator import upper_softmax


class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),

            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),

            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),

            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)

        )
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class GeneratorOneChannelResidual(AbstractGenerator):

    def __init__(self, latent_size):
        super(GeneratorOneChannelResidual, self).__init__()

        self._noise_dim = torch.tensor([latent_size])

        # Initializing layers with residual blocks
        self.layers = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size),
            nn.BatchNorm1d(2 * latent_size),
            nn.LeakyReLU(0.2),

            nn.Linear(2 * latent_size, 4* latent_size),
            nn.BatchNorm1d(4 * latent_size),
            nn.LeakyReLU(0.2),

            ResidualBlock2(4 * latent_size, 4 * latent_size),

            nn.Linear(4 * latent_size, 8 * latent_size),
            nn.BatchNorm1d(8 * latent_size),
            nn.LeakyReLU(0.2),

            ResidualBlock2(8 * latent_size, 8 * latent_size),

            nn.Linear(8 * latent_size, 32*32),
            upper_softmax()
        )

    def forward(self, x):
        return self.layers(x)

    def sample_subspace_masks(self, noise):
        return self.forward(noise).repeat(1, 3).view(-1, 3, 32, 32)