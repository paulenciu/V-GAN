import torch
from torch import nn

from src.models.Generator import upper_softmax
from src.models.generator.IGenerator import IGenerator


class GeneratorThreeChannel(IGenerator):
    def __init__(self, latent_size, img_size):
        self.noise_dim = torch.tensor([latent_size])
        super(GeneratorThreeChannel, self).__init__()

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
            nn.Linear(16 * latent_size, img_size),
            upper_softmax()
        )

    def forward(self, x):
        x = self.hidden(x)
        return x

    def get_noise_tensor_shape(self):
        return self.noise_dim

    def sample_subspace_masks(self, noise):
        return self.forward(noise).view(noise.shape[0], 3, 32, 32)