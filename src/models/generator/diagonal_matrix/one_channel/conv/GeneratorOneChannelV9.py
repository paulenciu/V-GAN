import torch
import numpy as np
from torch import nn
from src.models.generator.AbstractGenerator import AbstractGenerator


class GeneratorOneChannelV9DBN(AbstractGenerator):

    def __init__(self, latent_size, image_shape, initial_temperature=1.0, min_temperature=0.1, anneal_rate=0.0003):
        super().__init__()
        self._noise_dim = torch.tensor([latent_size])
        self._img_shape = image_shape

        # Temperature for slope annealing trick
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate  # Controls how ast the Temperature decreases

        self.initial = nn.Sequential(
            nn.Linear(latent_size, 256 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 7, 7))
        )

        self.generator = nn.Sequential(
            self._conv_transpose_block(256, 128, 4, 2, 1),  # 7x7 -> 14x14
            self._conv_transpose_block(128, 64, 4, 2, 1),   # 14x14 -> 28x28
            self._conv_transpose_block(64, 32, 4, 2, 1),    # 28x28 -> 56x56
            self._conv_transpose_block(32, 16, 4, 2, 1),    # 56x56 -> 112x112
            self._conv_transpose_block(16, 8, 4, 2, 1),     # 112x112 -> 224x224
            nn.Conv2d(8, 1, kernel_size=3, padding=1)      # Final output layer
        )

    def _conv_transpose_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def binarize_ste(self, x):
        """
        Binarization using the Straight-Through Estimator (STE) with slope annealing.
        """
        if self.training:
            T = max(self.min_temperature, self.temperature)
            probs = torch.sigmoid(x / T)
            return probs
        else:
            return (torch.sigmoid(x) > 0.5).float()

    def forward(self, noise, mode="train"):
        x = self.initial(noise)
        x = self.generator(x)
        x = self.binarize_ste(x)
        return x

    def sample_subspace_masks(self, noise, mode="train"):
        if mode == "train":
            self.train()
            return self.forward(noise, mode).repeat(1, self._img_shape[0], 1, 1)
        self.eval()
        return self.forward(noise, mode).repeat(1, self._img_shape[0], 1, 1)


    def anneal_temperature(self):
        """
        Gradually decreases the temperature for the slope annealing trick.
        """
        self.temperature = max(self.min_temperature, self.temperature - self.anneal_rate)
