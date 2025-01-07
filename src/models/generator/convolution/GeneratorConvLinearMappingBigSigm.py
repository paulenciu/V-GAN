import torch
from torch import nn

from src.models.generator.AbstractGenerator import AbstractGenerator

class GeneratorConvLinearMappingBigSigm(AbstractGenerator):

    def __init__(self, latent_size=None):
        super(GeneratorConvLinearMappingBigSigm, self).__init__()

        if latent_size is None:
            self._noise_dim = torch.tensor([1, 1, 1])
        else:
            self._noise_dim = latent_size

        self.main = nn.Sequential(
            # Input: (batch, 1, 1, 1)

            # 1) First Transpose Convolution: (1x1) -> (4x4)
            nn.ConvTranspose2d(1, 64, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Mean Pooling: (4x4) -> (2x2)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2) Second Transpose Convolution: (2x2) -> (8x8) by using stride=4
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Mean Pooling: (8x8) -> (4x4)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3) Third Transpose Convolution: (4x4) -> (16x16) using stride=4 again
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 4) Fourth Transpose Convolution: (16x16) -> (32x32) using stride=2, pad=1
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 5) Output Layer: (32x32) -> (32x32), single-channel
            nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    def sample_subspace_masks(self, noise):
        return self.forward(noise).repeat(1, 3, 1, 1)
