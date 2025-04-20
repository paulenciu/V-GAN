import torch
import math
import torch.nn as nn

from src.models.generator.AbstractGenerator import AbstractGenerator
from src.models.Generator import UpperSoftmax1D
from src.models.generator.modules.GaussianNoise import GaussianNoise
from torch.nn import Sigmoid, Dropout, BatchNorm1d
from torch.nn.utils.parametrizations import spectral_norm


class GeneratorOneChannelCNNBig(AbstractGenerator):

    def __init__(self, latent_size, image_shape):
        super().__init__()
        self._noise_dim = torch.tensor([latent_size, 1, 1])
        self._img_shape = image_shape
        #h, w = image_shape[1], image_shape[2]
        #assert h == w, "Image must be square"

        self.init_size = None
        self.linear = None
        self.conv_blocks = None
        self.dcgan(latent_size)
        #self._build_64x64_us(latent_size)
        # if h == 32:
        #     self._build_32x32(latent_size)
        # elif h == 28:
        #     self._build_28x28(latent_size)
        # elif h == 64:
        #
        # else:
        #     raise ValueError(f"Unsupported image size: {h}x{w}")

        self.upper_softmax = UpperSoftmax1D()
        self.softmax = nn.Softmax(dim=1)

    def _build_32x32(self, latent_size):
        self.init_size = 4
        self.linear = nn.Linear(latent_size, 512 * self.init_size ** 2)
        self.conv_blocks = nn.Sequential(
            GaussianNoise(stddev=0.1),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4->8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8->16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16->32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def _build_28x28(self, latent_size):
        self.init_size = 7
        self.linear = nn.Linear(latent_size, 256 * self.init_size ** 2)
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),  # 7->14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),  # 14->28
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def _build_64x64_us(self, latent_size):
        self.init_size = 4
        #self.linear = nn.Linear(latent_size, 512 * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            # Stage 1: 4x4 -> 8x8
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.utils.spectral_norm(nn.Conv2d(512, 256, kernel_size=3, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # Stage 2: 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.utils.spectral_norm(nn.Conv2d(256, 128, kernel_size=3, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # Stage 3: 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.utils.spectral_norm(nn.Conv2d(128, 64, kernel_size=3, padding=1)),
            nn.BatchNorm2d(64),
            #nn.LeakyReLU(0.2),

            # Stage 4: 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.utils.spectral_norm(nn.Conv2d(64, 32, kernel_size=3, padding=1)),
            nn.BatchNorm2d(32),
            #nn.LeakyReLU(0.2),

            # Additional refinement blocks
            nn.utils.spectral_norm(nn.Conv2d(32, 16, kernel_size=3, padding=1)),
            GaussianNoise(stddev=0.1),
            nn.BatchNorm2d(16),
            #nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(16, 8, kernel_size=3, padding=1)),
            nn.BatchNorm2d(8),
            #nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(8, 4, kernel_size=3, padding=1)),
            nn.BatchNorm2d(4),
            #nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(4, 2, kernel_size=3, padding=1)),
            nn.BatchNorm2d(2),
            #nn.LeakyReLU(0.2),

            # Final output (no BN/activation)
            nn.utils.spectral_norm(nn.Conv2d(2, 1, kernel_size=3, padding=1)),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(64 * 64 * 1, 64 * 64 * 1),
            # spectral_norm(
            #     nn.Linear(64 * 64 * 1, 64 * 64 * 1)
            # ),
            # GaussianNoise(stddev=0.1),
            # #nn.LeakyReLU(),
            # #nn.BatchNorm1d(64 * 64 * 1),
            # spectral_norm(
            #     nn.Linear(64 * 64 * 1, 64 * 64 * 1)
            # )
        )

    def dcgan(self, conv_dim=32, z_dim=100):

        #self.linear = nn.Linear(z_dim, 8 * 32 * self.init_size * self.init_size)

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
            # nn.Sigmoid()  # Output in [0,1]
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.output_layer = nn.Sequential(
            spectral_norm(nn.Linear(64 * 64 * 1, self._img_shape[1]*self._img_shape[2])),
            nn.Tanh()
        )



    def _build_64x64(self, latent_size):
        self.init_size = 4
        self.linear = nn.Linear(latent_size, 512 * self.init_size ** 2)
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4->8
            GaussianNoise(stddev=0.1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8->16
            GaussianNoise(stddev=0.1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16->32
            GaussianNoise(stddev=0.1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),  # 32->64
        )

    def forward(self, input, mode="train"):
        #x = self.linear(input)
        #x = x.view(x.size(0), -1, self.init_size, self.init_size)
        x = self.conv_blocks(input)
        x = self.output_layer(x.view(x.size(0), -1)).view(-1, 1, self._img_shape[1], self._img_shape[2])
        return x  # (batch, 1, H, W)


    def sample_subspace_masks(self, noise, mode="train"):
        if mode == "train":
            self.train()
            x = self.forward(noise, mode)

            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)
            x_flat = self.softmax(x_flat)
            #x = x_flat.view(batch_size, *self._img_shape)
            #x = x_flat.repeat(1,3).view(batch_size, *self._img_shape)
            return x_flat.view(x.size(0), -1)
        else:
            self.eval()
            with torch.no_grad():
                x = self.forward(noise, mode)
                batch_size = x.shape[0]
                x_flat = x.view(batch_size, -1)
                x_flat = self.upper_softmax(x_flat)
                x = torch.greater(x_flat, 1 / x_flat.shape[1])
                #x = x.view(batch_size, *self._img_shape)
                #x = x.repeat(1,3).view(batch_size, *self._img_shape)
            return x.view(x.size(0), -1)