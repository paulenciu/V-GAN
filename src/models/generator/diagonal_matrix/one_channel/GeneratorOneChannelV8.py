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

        self.initial = nn.Sequential(
            nn.Linear(latent_size, 512 * 7 * 7),
            nn.Unflatten(1, (512, 7, 7)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.upscale = nn.Sequential(
            # Stage 1: 7x7 -> 14x14
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256),

            # Stage 2: 14x14 -> 28x28
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),
            SelfAttention(128),  # Add attention at 28x28 resolution

            # Stage 3: 28x28 -> 56x56
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),

            # Stage 4: 56x56 -> 112x112
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 5: 112x112 -> 224x224
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.final = nn.Sequential(
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input, mode="train"):
        x = self.initial(input)
        x = self.upscale(x)
        return self.final(x)

    def sample_subspace_masks(self, noise, mode="train"):
        return self.forward(noise, mode).repeat(1, self._img_shape[0], 1, 1)