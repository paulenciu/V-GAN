import torch
import torch.nn as nn
from src.models.Generator import UpperSoftmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator
from src.models.generator.modules.GaussianNoise import GaussianNoise
from torch.nn.utils import spectral_norm


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class GeneratorOneChannelConv(AbstractGenerator):
    def __init__(self, latent_size, image_shape):
        super().__init__()
        C, H, W = image_shape
        self.latent_dim = latent_size
        self._noise_dim = torch.tensor([latent_size])
        self._img_shape = image_shape
        self.output_shape = image_shape
        self.num_elements = H * W

        self.initial_channels = 512
        current_channels = self.initial_channels
        current_size = 4

        # Build upsampling layers
        self.upsample_layers = nn.ModuleList()
        while current_size < min(H, W):
            self.upsample_layers.append(
                nn.Sequential(
                    spectral_norm(nn.ConvTranspose2d(
                        current_channels,
                        current_channels // 2,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    )),
                    GaussianNoise(stddev=0.1),
                    nn.BatchNorm2d(current_channels // 2),
                    nn.LeakyReLU(0.8)
                )
            )
            current_channels = current_channels // 2
            current_size *= 2

        # Use original 512 channels for initial projection
        self.init_proj = nn.Sequential(
            spectral_norm(nn.Linear(latent_size, self.initial_channels * 4 * 4)),
            GaussianNoise(stddev=0.1),
            nn.BatchNorm1d(self.initial_channels * 4 * 4),
            nn.LeakyReLU(0.8),
            Reshape(self.initial_channels, 4, 4)  # Now matches first layer's input
        )

        # Final conv uses accumulated channel count
        self.final_conv = spectral_norm(nn.Conv2d(current_channels, 1, kernel_size=3, padding=1))

        # Rest of the code remains the same...

        self.upper_softmax = UpperSoftmax1D()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        x = self.init_proj(z)
        for layer in self.upsample_layers:
            x = layer(x)
        x = self.final_conv(x)  # Shape: (batch, H*W, H', W')
        return x

    def sample_subspace_masks(self, noise, mode="train"):
        logits = self.forward(noise)
        batch_size = logits.size(0)

        flat_logits = logits.view(batch_size, -1)

        if mode == "train":
            probs = self.softmax(flat_logits)
            return probs.repeat(1, self._img_shape[0]).view(-1, *self._img_shape)
        else:
            probs = self.upper_softmax(flat_logits)
            threshold = 1 / self.num_elements
            mask = (probs > threshold).float()
            return mask.repeat(1, self._img_shape[0]).view(-1, *self._img_shape)