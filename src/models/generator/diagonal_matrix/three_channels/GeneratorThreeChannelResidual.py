import torch
from torch import nn

from src.models.Generator import UpperSoftmax2D
from src.models.generator.AbstractGenerator import AbstractGenerator

class GeneratorThreeChannelResidual(AbstractGenerator):

    def __init__(self, latent_size, depth=6, dropout=0.2):
        """
        Initialize the Generator with options for depth and dropout.

        :param latent_size: Size of the latent space.
        :param depth: Number of hidden layers to scale complexity.
        :param dropout: Dropout probability for regularization.
        """
        super(GeneratorThreeChannelResidual, self).__init__()

        self._noise_dim = torch.tensor([latent_size])

        layers = []
        input_dim = latent_size

        for i in range(depth):
            output_dim = input_dim * 2
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        # Final output layer
        layers.append(nn.Linear(input_dim, 32 * 32 * 3))
        layers.append(UpperSoftmax2D())

        # Build the model as a sequential block
        self.hidden = nn.Sequential(*layers)

        # Residual connections to refine generated features
        self.residual = nn.Sequential(
            nn.Linear(latent_size, 32 * 32 * 3),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass for the generator with residual connection.

        :param x: Input noise.
        :return: Generated output in the shape of [batch_size, 3, 32, 32].
        """
        base_output = self.hidden(x)
        residual_output = self.residual(x)
        return base_output + residual_output

    def sample_subspace_masks(self, noise):
        """
        Generate masks from the latent noise input.

        :param noise: Latent noise.
        :return: Reshaped generated cifar10 as [batch_size, 3, 32, 32].
        """
        return self.forward(noise).view(noise.shape[0], 3, 32, 32)
