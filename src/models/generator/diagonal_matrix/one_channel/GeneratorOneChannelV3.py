import torch
from torch import nn

from src.models.Generator import upper_softmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator


class GeneratorOneChannelV3(AbstractGenerator):

    def __init__(self, latent_size, img_shape):
        super(GeneratorOneChannelV3, self).__init__()

        self._noise_dim = torch.tensor([latent_size])
        self._img_shape = img_shape

        img_size = img_shape[1]

        self.layers = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size),
            nn.BatchNorm1d(2 * latent_size),
            nn.LeakyReLU(0.2),

            nn.Linear(2 * latent_size, 4 * latent_size),
            nn.BatchNorm1d(4 * latent_size),
            nn.LeakyReLU(0.2),

            nn.Linear(4 * latent_size, 8 * latent_size),
            nn.BatchNorm1d(8 * latent_size),
            nn.LeakyReLU(0.2),

            nn.Linear(8 * latent_size, img_size * img_size),
        )
        self.upper_softmax = upper_softmax1D()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, mode="train"):
        x = self.layers(input)
        if mode == "train":
            return self.softmax(x)

        return self.upper_softmax(x)

    def sample_subspace_masks(self, noise, mode="train"):
        x = self.forward(noise, mode).repeat(1, self.img_shape[0])
        x = x.view(-1, self.img_shape[0], self.img_shape[1], self.img_shape[2])
        return x