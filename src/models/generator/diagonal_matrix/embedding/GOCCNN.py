import torch
import math
import torch.nn as nn

from src.models.generator.AbstractGenerator import AbstractGenerator
from src.models.Generator import UpperSoftmax1D
from src.models.generator.modules.GaussianNoise import GaussianNoise
from torch.nn import Sigmoid, Dropout, BatchNorm1d
from torch.nn.utils.parametrizations import spectral_norm


class GOCCNN(AbstractGenerator):

    def __init__(self, latent_size, image_shape):
        super().__init__()
        self._noise_dim = torch.tensor([latent_size, 1, 1])
        self._img_shape = image_shape
        self.img_size = math.prod(self._img_shape)

        self.init_size = None
        self.linear = None
        self.conv_blocks = None
        self._build(latent_size)

        self.upper_softmax = UpperSoftmax1D()
        self.softmax = nn.Softmax(dim=1)


    def _build(self, conv_dim=32, z_dim=100):

        self.conv_blocks = nn.Sequential(

            # Input layer (z_dim -> conv_dim*8)
            nn.ConvTranspose2d(z_dim, conv_dim * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(conv_dim * 8),
            nn.ReLU(),

            # Layer 1: 4x4 -> 8x8
            nn.ConvTranspose2d(conv_dim * 8, conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(),

            # Layer 2: 8x8 -> 16x16
            nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(),

            # Layer 3: 16x16 -> 32x32
            nn.ConvTranspose2d(conv_dim * 2, conv_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),

            # Output layer: 32x32 -> 64x64
            nn.ConvTranspose2d(conv_dim, 1, kernel_size=4, stride=2, padding=1, bias=False),

            #SMOOTHING
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(64 * 64 * 1, self.img_size),
            nn.Tanh()
        )

    def forward(self, input, mode="train"):

        x = self.conv_blocks(input)
        x = self.output_layer(x.view(x.size(0), -1))
        return x


    def sample_subspace_masks(self, noise, mode="train"):
        if mode == "train":
            self.train()
            x = self.forward(noise, mode)

            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)
            x_flat = self.softmax(x_flat)
            return x_flat
        else:
            self.eval()
            with torch.no_grad():
                x = self.forward(noise, mode)
                batch_size = x.shape[0]
                x_flat = x.view(batch_size, -1)
                x_flat = self.upper_softmax(x_flat)
                x = torch.greater(x_flat, 1 / x_flat.shape[1])
            return x