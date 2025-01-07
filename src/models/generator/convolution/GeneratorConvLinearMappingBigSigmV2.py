from typing import Optional

import torch
from torch import nn

from src.models.generator.AbstractGenerator import AbstractGenerator


class GeneratorConvLinearMappingBigSigmV2(AbstractGenerator):

    def __init__(self, latent_size: Optional[torch.Tensor]=None):
        super(GeneratorConvLinearMappingBigSigmV2, self).__init__()

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
            nn.Sigmoid()  # Mask values between 0 and 1
        )

    def forward(self, input):
        return self.main(input)

    def sample_subspace_masks(self, noise):
        return self.forward(noise).repeat(1, 3, 1, 1)  # Repeat across channels if needed
