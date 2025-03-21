from cmath import log

import torch
import numpy as np
from src.models.generator.modules.GaussianNoise import GaussianNoise
from torch import nn

from src.models.Generator import UpperSoftmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator
from src.models.generator.modules.BatchDiscrimination import BatchDiscrimination


class GeneratorOneChannelSNBD(AbstractGenerator):

    def __init__(self, latent_size, image_shape):
        super().__init__()

        if isinstance(latent_size, torch.Tensor):
            latent_size = latent_size.item()

        self._noise_dim = torch.tensor([latent_size])
        self._img_shape = image_shape

        img_size = image_shape[2] * image_shape[2]
        rel_size = int(img_size / latent_size)
        self.latent_size = latent_size
        self.img_size = img_size
        amount_layers = 6
        self.increase = log(rel_size, amount_layers).real

        layers = [self.get_layer(layer) for layer in range(1, amount_layers)]
        layers += [self.get_layer(amount_layers, last=True)]
        self.layers = nn.Sequential(*layers)
        self.upper_softmax = UpperSoftmax1D()
        self.softmax = nn.Softmax(dim=1)

    def get_layer(self, layer: int, last=False):
        input_size = round(pow(self.increase, layer - 1) * self.latent_size)
        output_size = round(pow(self.increase, layer) * self.latent_size)

        layer = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Linear(input_size, output_size)
            ),
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
        )

        last_layer = nn.Sequential(
            nn.Linear(input_size, self.img_size),
        )

        return last_layer if last else layer

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
