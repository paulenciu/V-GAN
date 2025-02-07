from typing import Optional

import torch
from src.models.Generator import UpperSoftmax2D, UpperSparsemax2D
from torch import nn
import torch.nn.functional as F

from src.models.generator.AbstractGenerator import AbstractGenerator


class GeneratorConvLinearMappingBigSoftmax(AbstractGenerator):

    def __init__(self, latent_size: Optional[torch.Tensor]=None):
        super(GeneratorConvLinearMappingBigSoftmax, self).__init__()

        if latent_size is None:
            self._noise_dim = torch.tensor([64, 1, 1])
        else:
            self._noise_dim = latent_size

        latent_size = self._noise_dim[0].item()

        self.main = nn.Sequential(
            # Input: (batch, noise_dim, 1, 1)

            # Project to 4x4x512
            nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample to 8x8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample to 16x16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample to 32x32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Final convolution to get 1 output channel (32x32)
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.upper_softmax = UpperSoftmax2D()
        self.sparse_max = UpperSparsemax2D()

    def forward(self, input, mode="train"):
        x = self.main(input)
        if mode == "train":
            return self.softmax(x)
        return self.sparse_max(x)

    def stable_softmax(self, x, dim=1):
        # Subtract the maximum value for numerical stability
        x_max = torch.max(x, dim=dim, keepdim=True).values
        x_stable = x - x_max
        # Compute the softmax
        return F.softmax(x_stable, dim=dim)

    def softmax(self, x):
        x_flattened = x.view(x.size(0), -1)
        x_flattened = self.stable_softmax(x_flattened)
        x = x_flattened.view(x.size(0), x.size(1), x.size(2), x.size(3))
        return x

    def sample_subspace_masks(self, noise, mode="train"):
        return self.forward(noise, mode).repeat(1, 3, 1, 1)
