import torch
from src.models.generator.AbstractGenerator import AbstractGenerator
from torch import nn


class GeneratorOneChannelV6(AbstractGenerator):

    def __init__(self, latent_size, image_shape):
        super().__init__()
        self._noise_dim = torch.tensor([latent_size])
        self._img_shape = image_shape
        self.img_size = image_shape[1] * image_shape[2]  # Height x Width

        self.layers = nn.Sequential(
            *self._build_hidden_layers(latent_size),
            self._build_final_layer(),
            nn.Sigmoid()
        )

        # Add spatial awareness
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _build_hidden_layers(self, latent_size):
        layer_sizes = [
            latent_size,
            latent_size * 2,
            self.img_size // 4,
            self.img_size // 2
        ]

        return [
            nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            )
            for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
        ]

    def _build_final_layer(self):
        return nn.Linear(self.img_size // 2, self.img_size)

    def _init_weights(self):
        """Proper weight initialization for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, input, mode="train"):
        x = self.layers(input)

        x = x.view(-1, 1, self._img_shape[1], self._img_shape[2])
        x = self.spatial_attention(x)
        return x

    def sample_subspace_masks(self, noise, mode="train"):
        masks = self.forward(noise, mode)
        return masks.repeat(1, self._img_shape[0], 1, 1)  # Repeat across channels
