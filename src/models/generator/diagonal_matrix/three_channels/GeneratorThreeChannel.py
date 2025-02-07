import torch
from torch import nn

from src.models.Generator import UpperSoftmax2D, UpperSoftmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator


class GeneratorThreeChannel(AbstractGenerator):

    def __init__(self, latent_size, img_shape):
        super(GeneratorThreeChannel, self).__init__()

        self._noise_dim = torch.tensor([latent_size])
        self._img_shape = img_shape

        self.hidden = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size),
            nn.BatchNorm1d(2 * latent_size),
            nn.ReLU(),
            nn.Linear(2 * latent_size, 4 * latent_size),
            nn.BatchNorm1d(4 * latent_size),
            nn.ReLU(),
            nn.Linear(4 * latent_size, 8 * latent_size),
            nn.BatchNorm1d(8 * latent_size),
            nn.ReLU(),
            nn.Linear(8 * latent_size, 16 * latent_size),
            nn.BatchNorm1d(16 * latent_size),
            nn.ReLU(),
            nn.Linear(16 * latent_size, self._img_shape[0]*self._img_shape[1]*self._img_shape[2]),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.upper_softmax = UpperSoftmax1D()

    def forward(self, x, mode):
        x = self.hidden(x)

        if mode == "train":
            return self.softmax(x)

        return self.upper_softmax(x)

    def sample_subspace_masks(self, noise, mode="train"):
        return self.forward(noise, mode).view(noise.shape[0], self._img_shape[0],self._img_shape[1], self._img_shape[2])