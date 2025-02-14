import torch
from torch import nn

from src.models.generator.diagonal_matrix.three_channels.GeneratroOneChannelV10DBN import BatchDiscrimination
from src.models.generator.modules.GaussianNoise import GaussianNoise
from src.models.Generator import UpperSoftmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator
from src.models.generator.modules.SelfAttention import SelfAttention


class GeneratorThreeChannelV10DBNSA(AbstractGenerator):
    def __init__(self, latent_size, image_shape, initial_temperature=1.0,
                 min_temperature=0.1, anneal_rate=0.01, bd_out_features=100):
        super().__init__()
        self._noise_dim = torch.tensor([latent_size])
        self.latent_size = latent_size
        self.img_shape = image_shape
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate
        self.bd_out_features = bd_out_features

        self.init_proj = nn.Sequential(
            nn.Linear(latent_size, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.LeakyReLU(0.2)
        )

        # We define the ConvBlocks
        self.block1 = ConvBlock(512, 256, input_spatial_size=(4, 4),
                                bd_out_features=self.bd_out_features)
        self.block2 = ConvBlock(256, 128, input_spatial_size=(8, 8),
                                bd_out_features=self.bd_out_features)

        # Insert a self-attention layer after the second block (this acts on 16x16)
        self.self_attention1 = SelfAttention(128)

        self.block3 = ConvBlock(128, 64, input_spatial_size=(16, 16),
                                bd_out_features=self.bd_out_features)

        self.self_attention2 = SelfAttention(64)

        # Final convolution to produce RGB (or whatever # of channels is in image_shape)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(64, image_shape[0], kernel_size=3, stride=1, padding=1),
        )

        self.softmax = nn.Softmax(dim=-1)

    def binarize_ste(self, x):
        if self.training:
            T = max(self.min_temperature, self.temperature)
            probs = torch.sigmoid(x / T)
            return probs
        else:
            discrete = (x > 0.5).float()
            return discrete

    def forward(self, z, mode="train"):
        if mode == "train":
            self.train()
        else:
            self.eval()

        x = self.init_proj(z)
        x = x.view(-1, 512, 4, 4)

        # Pass through your upsampling blocks
        x = self.block1(x)  # from 4x4 -> 8x8
        x = self.block2(x)  # from 8x8 -> 16x16

        # Self-attention at 16x16
        x = self.self_attention1(x)

        x = self.block3(x)  # from 16x16 -> 32x32


        x = self.to_rgb(x)  # (B, image_shape[0], 32, 32) for example


        x = x.view(x.shape[0], -1)
        x = self.softmax(x)
        if mode == "train":
            return x.view(-1, 3, 32, 32)
        return torch.greater_equal(x, 1 / x.shape[0]).view(-1, 3, 32, 32)

    def sample_subspace_masks(self, noise, mode="train"):
        masks = self.forward(noise, mode)
        return masks.view(-1, *self.img_shape)

    def anneal_temperature(self):
        self.temperature = max(self.min_temperature, self.temperature - self.anneal_rate)
        print(f"New temperature: {self.temperature}")

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_spatial_size, bd_out_features=100):
        super().__init__()
        self.in_channels = in_channels
        H, W = input_spatial_size
        self.in_features = in_channels * H * W
        self.bd = BatchDiscrimination(self.in_features, bd_out_features)
        self.conv_transpose = nn.utils.spectral_norm(
            nn.ConvTranspose2d(in_channels + 1, out_channels, 4, 2, 1, bias=False)
        )
        self.gaussian_noise = GaussianNoise(stddev=0.1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        batch_size = x.shape[0]
        spatial_size = x.shape[2:]
        x_flat = x.view(batch_size, -1)
        x_bd_out = self.bd(x_flat)
        x_bd = x_bd_out[:, :-1].view(batch_size, self.in_channels, *spatial_size)
        c_bd = x_bd_out[:, -1].view(batch_size, 1, 1, 1).expand(-1, -1, *spatial_size)
        x_concat = torch.cat([x_bd, c_bd], dim=1)
        x = self.conv_transpose(x_concat)
        x = self.gaussian_noise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x