from cmath import log

import torch
from torch import nn

from src.models.Generator import UpperSoftmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator


class GeneratorOneChannelV7(AbstractGenerator):

    def __init__(self, latent_size, image_shape):

        if isinstance(latent_size, torch.Tensor):
            latent_size = latent_size.item()

        super(GeneratorOneChannelV7, self).__init__()

        self._noise_dim = torch.tensor([latent_size])
        self._img_shape = image_shape

        self.model = nn.Sequential(
            #latent_dim → 256x7x7
            nn.Linear(latent_size, 256 * 7 * 7),
            nn.Unflatten(1, (256, 7, 7)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Upsample to 14x14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Upsample to 28x28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Upsample to 56x56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Upsample to 112x112
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Final layer: Upsample to 224x224 with 1 channel (grayscale)
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1, bias=False),  # Changed from 3→1
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def forward(self, input, mode="train"):
        x = self.model(input)
        return x

    def sample_subspace_masks(self, noise, mode="train"):
        return self.forward(noise, mode).repeat(1, self._img_shape[0], 1, 1)
