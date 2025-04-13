import torch
from src.models.Generator import UpperSoftmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator
from src.models.generator.modules.GaussianNoise import GaussianNoise
from torch import nn


class GeneratorRes50V2(AbstractGenerator):

    def __init__(self, latent_size, image_shape):
        super().__init__()

        if isinstance(latent_size, torch.Tensor):
            latent_size = latent_size.item()

        self._noise_dim = torch.tensor([latent_size])
        self.img_size = 4 * 512 * 7 * 7

        self.layers = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Linear(latent_size, 4 * latent_size),
            ),
            GaussianNoise(stddev=0.1),
            nn.BatchNorm1d(4 * latent_size),
            nn.GELU(),

            nn.utils.spectral_norm(
                nn.Linear(4 * latent_size, self.img_size),
            ),

        )
        self.upper_softmax = UpperSoftmax1D()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, mode="train"):
        x = self.layers(input)
        return x

    def sample_subspace_masks(self, noise, mode="train"):

        if mode == "train":
            self.train()
            x = self.forward(noise, mode)
            x = self.softmax(x)
            return x
        else:
            self.eval()
            x = self.forward(noise, mode)
            x = self.upper_softmax(x)
            x = torch.greater(x, 1 / x.shape[1])
            return x