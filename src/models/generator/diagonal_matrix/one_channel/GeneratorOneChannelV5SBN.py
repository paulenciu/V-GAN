from cmath import log

import numpy as np
import torch
from torch import nn

from src.models.Generator import UpperSoftmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator


class GeneratorOneChannelV5SBN(AbstractGenerator):

    def __init__(self, latent_size, image_shape):

        if isinstance(latent_size, torch.Tensor):
            latent_size = latent_size.item()

        super(GeneratorOneChannelV5SBN, self).__init__()

        self._noise_dim = torch.tensor([latent_size])
        self._img_shape = image_shape

        img_size = image_shape[2] * image_shape[2]
        rel_size = int(img_size/latent_size)
        self.latent_size = latent_size
        self.img_size = img_size
        amount_layers = 4
        self.increase = log(rel_size, amount_layers).real

        layers = [self.get_layer(layer) for layer in range(1, amount_layers)]
        layers += [self.get_layer(amount_layers, last=True)]
        self.layers = nn.Sequential(*layers)

    def get_layer(self, layer: int, last=False):
        input_size = round(pow(self.increase, layer - 1) * self.latent_size)
        output_size = round(pow(self.increase, layer) * self.latent_size)

        layer = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
        )
        last_layer = nn.Sequential(
            nn.Linear(input_size, self.img_size),
            nn.Sigmoid(),
        )

        return last_layer if last else layer

    def forward(self, input, mode="train"):
        x = self.layers(input)
        return x

    def sample_subspace_masks(self, noise, mode="train"):
        activation = self.forward(noise, mode).repeat(1, self._img_shape[0]).view(-1, *self._img_shape)
        if mode == "train":
            return activation

        gaussian_threshold = np.random.rand(1) #using np.random, as torch.rand 

        gaussian_threshold =  torch.from_numpy(gaussian_threshold).to("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        activation = torch.greater_equal(activation, gaussian_threshold)
        print("Gaussian threshold: ", gaussian_threshold)
        return activation