from cmath import log

import torch
import numpy as np
from src.models.generator.modules.GaussianNoise import GaussianNoise
from torch import nn

from src.models.Generator import UpperSoftmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator
from src.models.generator.modules.BatchDiscrimination import BatchDiscrimination


class GeneratorOneChannelV4DBN(AbstractGenerator):

    def __init__(self, latent_size, image_shape, initial_temperature=1.0, min_temperature=0.1, anneal_rate=0.01):
        super().__init__()
        self._noise_dim = torch.tensor([latent_size])
        self._img_shape = image_shape

        # Temperature for slope annealing trick
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate  # Controls how ast the Temperature decreases

        img_size = image_shape[2] * image_shape[2]
        rel_size = int(img_size/latent_size)
        self.latent_size = latent_size
        self.img_size = img_size
        amount_layers = 6
        self.increase = log(rel_size, amount_layers).real

        layers = [self.get_layer(layer) for layer in range(1, amount_layers)]
        layers += [self.get_layer(amount_layers, last=True)]
        self.layers = nn.Sequential(*layers)

    def get_layer(self, layer: int, last=False):
        input_size = round(pow(self.increase, layer - 1) * self.latent_size)
        output_size = round(pow(self.increase, layer) * self.latent_size)

        layer = nn.Sequential(
            #BatchDiscrimination(input_size, input_size),
            nn.utils.spectral_norm(
                nn.Linear(input_size+1, output_size)
            ),
            GaussianNoise(stddev=0.1),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.8),
        )

        last_layer = nn.Sequential(
            #nn.Linear(input_size + 1, self.img_size),
            nn.Linear(input_size, self.img_size),
            nn.Sigmoid(),
        )

        return last_layer if last else layer

    def binarize_ste(self, x):
        """
        Binarization using the Straight-Through Estimator (STE) with slope annealing.
        """
        if self.training:
            T = max(self.min_temperature, self.temperature)
            probs = torch.sigmoid(x / T)
            return probs
        else:
            print("-----------Sample----------")
            print(x[:3])
            discrete = (x > 0.5).float()
            print("-----------Sample Discrete----------")
            print(discrete[:3])
            return discrete

    def forward(self, input, mode="train"):
        x = self.layers(input)
        x = self.binarize_ste(x)
        return x

    def sample_subspace_masks(self, noise, mode="train"):
        if mode == "train":
            self.train()
        else:
            self.eval()
        return self.forward(noise, mode).repeat(1, self._img_shape[0]).view(-1, *self._img_shape)

    def anneal_temperature(self):
        """
        Gradually decreases the temperature for the slope annealing trick.
        """
        self.temperature = max(self.min_temperature, self.temperature - self.anneal_rate)
        print("New temperature: {}".format(self.temperature))