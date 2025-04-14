import math

import torch
from torch import nn

from src.models.Generator import UpperSoftmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator


class GeneratorOneChannelBig(AbstractGenerator):

    def __init__(self, latent_size, image_shape):
        super(GeneratorOneChannelBig, self).__init__()

        if isinstance(latent_size, torch.Tensor):
            latent_size = latent_size.item()

        self._noise_dim = torch.tensor([latent_size])
        self._img_shape = image_shape
        image_size = math.prod(self._img_shape[1:])

        self.layers = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size),
            nn.BatchNorm1d(2 * latent_size),
            nn.LeakyReLU(),
            nn.Linear(2*latent_size, 4 *latent_size),
            nn.BatchNorm1d(4 * latent_size),
            nn.LeakyReLU(),
            nn.Linear(4*latent_size, 8 * latent_size),
            nn.BatchNorm1d(8 * latent_size),
            nn.LeakyReLU(),
            nn.Linear(8 * latent_size, image_size),
        )
        self.upper_softmax = UpperSoftmax1D()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, mode="train"):
        x = self.layers(input)
        return x

    def sample_subspace_masks(self, noise, mode="train"):

        if mode == "train":
            self.train()
            x = self.forward(noise, mode)
            x = self.softmax(x)
            return x.repeat(1, self._img_shape[0]).view(-1, *self._img_shape)
        else:
            self.eval()
            x = self.forward(noise, mode)
            x = self.upper_softmax(x)
            x = torch.greater(x, 1 / x.shape[1])
            return x.repeat(1, self._img_shape[0]).view(-1, *self._img_shape)

