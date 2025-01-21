import torch
from torch import nn
from src.models.generator.AbstractGenerator import AbstractGenerator

class GeneratorOneChannelConv(AbstractGenerator):
    def __init__(self, latent_size, image_size=32, channels=3, feature_maps=64):
        """
        Generator that creates a diagonal matrix for a single channel and applies it
        across all channels (RGB).

        :param latent_size: Size of the latent space.
        :param image_size: Size of the diagonal matrix (assumed square, e.g., 32x32).
        :param channels: Number of output channels (e.g., 3 for RGB).
        :param feature_maps: Base number of feature maps for the first ConvTranspose layer.
        """
        super(GeneratorOneChannelConv, self).__init__()

        self._noise_dim = torch.tensor([latent_size])
        self.latent_size = latent_size
        self.image_size = image_size
        self.channels = channels
        self.feature_maps = feature_maps

        # Fully connected layer to project the latent space to an initial feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_size, feature_maps * 8 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsampling layers
        layers = []
        current_features = feature_maps * 8  # Start with the highest feature map count
        spatial_dim = 4  # Start with 4x4 spatial dimensions

        # Add 3 upsampling blocks to reach 32x32
        while spatial_dim < image_size:
            next_features = current_features // 2
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=current_features,
                    out_channels=next_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(next_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_features = next_features
            spatial_dim *= 2

        # Final layer to produce a single-channel output
        layers.append(
            nn.Conv2d(
                in_channels=current_features,
                out_channels=1,  # Single channel for diagonal matrix
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        )
        layers.append(nn.Tanh())  # Output in [-1, 1]

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the generator.

        :param x: Latent vector of shape [batch_size, latent_size].
        :return: Generated diagonal matrix of shape [batch_size, 3, 32, 32].
        """
        # Project latent vector to 4x4 feature map
        x = self.fc(x)
        x = x.view(-1, self.feature_maps * 8, 4, 4)  # Reshape to [batch_size, features, 4, 4]

        # Upsample to [batch_size, 1, 32, 32]
        single_channel_output = self.conv_layers(x)

        # Expand the single channel to RGB by repeating across 3 channels
        return single_channel_output.repeat(1, 3, 1, 1)

    def sample_subspace_masks(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generate subspace masks by sampling from the generator.

        :param noise: Input noise tensor of shape [batch_size, latent_size].
        :return: Generated masks of shape [batch_size, 3, 32, 32].
        """
        return self.forward(noise)
