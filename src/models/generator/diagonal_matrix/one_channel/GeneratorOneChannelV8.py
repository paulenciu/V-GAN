import torch
from torch import nn
from src.models.Generator import upper_softmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H * W)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value(x).view(B, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x


class GeneratorOneChannelV8(AbstractGenerator):
    def __init__(self, latent_size, image_shape):
        super().__init__()
        self._noise_dim = torch.tensor([latent_size])
        self._img_shape = image_shape

        # Reduced initial channels and optimized block structure
        self.initial = nn.Sequential(
            nn.Linear(latent_size, 256 * 7 * 7),  # Reduced from 512
            nn.Unflatten(1, (256, 7, 7)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.upscale = nn.Sequential(
            # Stage 1: 7x7 -> 14x14
            self._upscale_block(256, 128),  # Reduced channels
            ResidualBlock(128),

            # Stage 2: 14x14 -> 28x28
            self._upscale_block(128, 64),
            ResidualBlock(64),
            SelfAttention(64),  # Lower dimension attention

            # Stage 3: 28x28 -> 56x56
            self._upscale_block(64, 32),
            ResidualBlock(32),

            # Stage 4: 56x56 -> 112x112
            self._upscale_block(32, 16),

            # Stage 5: 112x112 -> 224x224
            self._upscale_block(16, 8),
        )

        self.final = nn.Sequential(
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def _upscale_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input, mode="train"):
        x = self.initial(input)
        x = self.upscale(x)
        return self.final(x)

    def sample_subspace_masks(self, noise, mode="train"):
        return self.forward(noise, mode).repeat(1, self._img_shape[0], 1, 1)