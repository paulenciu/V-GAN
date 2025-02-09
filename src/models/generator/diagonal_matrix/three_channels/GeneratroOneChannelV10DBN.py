import torch
from torch import nn
from src.models.generator.modules.GaussianNoise import GaussianNoise
from src.models.Generator import UpperSoftmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator

import torch
from torch import nn


class BatchDiscrimination(nn.Module):

    def __init__(self, in_features, out_features):
        super(BatchDiscrimination, self).__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features))  # Learnable projection matrix

    def forward(self, x):

        # Project features into a new space
        M = x @ self.T  # (batch_size, out_features)

        # Compute pairwise L1 distance
        M_exp = M.unsqueeze(0)  # (1, batch_size, out_features)
        M_exp_T = M.unsqueeze(1)  # (batch_size, 1, out_features)
        dist = torch.abs(M_exp - M_exp_T).sum(dim=2)  # (batch_size, batch_size)

        # Apply exponential kernel to encourage diversity
        exp_kernel = torch.exp(-dist)

        # Mean feature vector for each sample
        batch_features = exp_kernel.mean(dim=1, keepdim=True)  # (batch_size, 1)

        # Concatenate batch features to input
        x = torch.cat([x, batch_features], dim=1)  # (batch_size, in_features + 1)
        return x

class GeneratorThreeChannelV10DBN(AbstractGenerator):

    def __init__(self, latent_size, image_shape, initial_temperature=1.0, min_temperature=0.1, anneal_rate=0.01):
        super().__init__()

        self._noise_dim = torch.tensor([latent_size])

        self.latent_size = latent_size
        self.img_shape = image_shape  # (channels, height, width)
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate

        # Initial projection to 4x4 feature map
        self.init_proj = nn.Sequential(
            nn.Linear(latent_size, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.LeakyReLU(0.2)
        )

        # Convolutional blocks
        self.layers = nn.Sequential(
            ConvBlock(512, 256),  # 4x4 -> 8x8
            ConvBlock(256, 128),  # 8x8 -> 16x16
            ConvBlock(128, 64),  # 16x16 -> 32x32
            nn.Conv2d(64, image_shape[0], 3, 1, 1),
            nn.Sigmoid()
        )

    def binarize_ste(self, x):
        """Straight-Through Estimator with slope annealing."""
        if self.training:
            T = max(self.min_temperature, self.temperature)
            probs = torch.sigmoid(x / T)
            return probs
        else:
            print("-----------Sample----------")
            print(x[:3])

            discrete = (x > 0.5).float()

            print("-----------Sample Discrete----------")
            print(discrete[:3])
            return discrete

    def forward(self, z, mode="train"):

        if mode == "train":
            self.train()
        else:
            self.eval()

        x = self.init_proj(z)
        x = x.view(-1, 512, 4, 4)
        x = self.layers(x)
        return self.binarize_ste(x)

    def sample_subspace_masks(self, noise, mode="train"):
        masks = self.forward(noise, mode)
        return masks.view(-1, *self.img_shape)

    def anneal_temperature(self):
        self.temperature = max(self.min_temperature, self.temperature - self.anneal_rate)
        print(f"New temperature: {self.temperature}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_size = 4
        self.block = nn.Sequential(
            # Batch discrimination on flattened features
            BatchDiscrimination(in_channels * kernel_size ** 2 , in_channels * kernel_size ** 2),

            # Transposed convolution with spectral norm
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(in_channels + 1, out_channels, kernel_size, 2, 1, bias=False)
                #nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)

            ),
            GaussianNoise(stddev=0.1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # Flatten spatial dimensions for batch discrimination
        batch_size = x.shape[0]
        spatial_size = x.shape[2:]
        x_flat = x.view(batch_size, -1)

        # Apply batch discrimination
        x_bd = self.block[0](x_flat)
        x_bd, c_bd = x_bd[:, :-1], x_bd[:, -1]
        x_bd = x_bd.view(batch_size, -1, *spatial_size)

        c_bd = c_bd.view(500, 1, 1, 1).expand(-1, -1, 4, 4)  # Shape: [500, 1, 4, 4]

        # Step 2: Concatenate along the channel dimension (dim=1)
        x_bd = torch.cat([x_bd, c_bd], dim=1)

        # Process through remaining layers
        return self.block[1:](x_bd)