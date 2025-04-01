import torch
import torch.nn as nn
from src.models.Generator import UpperSoftmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator
from src.models.generator.modules.GaussianNoise import GaussianNoise
from torch.nn.utils import spectral_norm


class GeneratorRes50ConvV2(AbstractGenerator):
    def __init__(self, latent_size, output_shape=(2048, 7, 7)):
        super().__init__()
        self.latent_dim = latent_size
        self.output_shape = output_shape
        self.output_shape = (2048, 7, 7)

        self.num_elements = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]

        self._noise_dim = torch.tensor([latent_size])
        self._img_shape = (self.num_elements, 1)

        # Initial dense projection
        self.init_proj = nn.Sequential(
            spectral_norm(nn.Linear(latent_size, 512 * 4 * 4)),
            GaussianNoise(stddev=0.1),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.LeakyReLU(0.8)
        )

        # Convolutional blocks
        self.conv_blocks = nn.Sequential(
            Reshape(512, 4, 4),

            spectral_norm(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0)),
            GaussianNoise(stddev=0.1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.8),

            spectral_norm(nn.Conv2d(256, 128, 3, padding=1)),
            GaussianNoise(stddev=0.1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.8),


            nn.Conv2d(128, self.output_shape[0], 3, padding=1)
        )

        self.upper_softmax = UpperSoftmax1D()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        x = self.init_proj(z)
        x = self.conv_blocks(x)
        return x

    def sample_subspace_masks(self, noise, mode="train"):
        logits = self.forward(noise)
        batch_size = logits.size(0)

        flat_logits = logits.view(batch_size, -1)

        if mode == "train":
            probs = self.softmax(flat_logits)
        else:
            probs = self.upper_softmax(flat_logits)
            threshold = 1 / self.num_elements
            mask = (probs > threshold).float()
            return mask

        return probs


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)