import torch
from torch import nn

from src.models.Generator import upper_softmax2D
from src.models.generator.AbstractGenerator import AbstractGenerator

class GeneratorOneChannelResidualBigInv(AbstractGenerator):

    def __init__(self, latent_size, depth=5, dropout=0.2):
        """
        Initialize the Generator with options for depth and dropout.

        :param latent_size: Size of the latent space.
        :param depth: Number of hidden layers to scale complexity.
        :param dropout: Dropout probability for regularization.
        """
        super(GeneratorOneChannelResidualBigInv, self).__init__()

        self._noise_dim = torch.tensor([latent_size])

        layers = []
        input_dim = latent_size

        for i in range(depth):
            output_dim = input_dim * 2
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim))  # Stabilized BatchNorm
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # Avoid ReLU dead neurons
            layers.append(nn.Dropout(dropout))  # Regularization
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, 32 * 32))

        self.hidden = nn.Sequential(*layers)
        self.residual = nn.Sequential(
            nn.Linear(latent_size, 32 * 32)
        )

        self.upper_softmax = upper_softmax2D()

    def forward(self, x):
        base_output = self.hidden(x)
        residual_output = self.residual(x)
        return nn.functional.softmax(base_output + residual_output, dim=1)


    def sample_subspace_masks(self, noise):
        return self.forward(noise).repeat(1, 3).view(-1, 3, 32, 32)
