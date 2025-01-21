import torch
from torch import nn

from src.models.Generator import upper_softmax2D
from src.models.generator.AbstractGenerator import AbstractGenerator


class GeneratorOneChannelResidualV2Inv(AbstractGenerator):

    def __init__(self, latent_size):
        super(GeneratorOneChannelResidualV2Inv, self).__init__()

        self._noise_dim = torch.tensor([latent_size])

        self.layers_in = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2 * latent_size),

            nn.Linear(2 * latent_size, 4 * latent_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(4 * latent_size),

            nn.Linear(4 * latent_size, 8 * latent_size),
            nn.BatchNorm1d(8 * latent_size),
            nn.LeakyReLU(0.2),


            nn.Linear(8 * latent_size, 8 * latent_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(8 * latent_size)
        )

        self.layers_mid = nn.Sequential(
            nn.Linear(8 * latent_size, 8 * latent_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(8 * latent_size),

            nn.Linear(8 * latent_size, 8 * latent_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(8 * latent_size),

            nn.Linear(8 * latent_size, 8 * latent_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(8 * latent_size),
        )

        self.layers_out =  nn.Sequential(
            nn.Linear(8 * latent_size, 32*32),
            upper_softmax2D()
        )

    def forward(self, input):
        x1 = self.layers_in(input)
        x2 = self.layers_mid(x1) + x1
        return self.layers_out(x2)

    def sample_subspace_masks(self, noise):
        return self.forward(noise).repeat(1, 3).view(-1, 3, 32, 32)