import torch
from torch import nn


class GeneratorConvLinearMappingBig(nn.Module):
    def __init__(self):
        super(GeneratorConvLinearMappingBig, self).__init__()
        self.noise_dim = torch.tensor([1, 1, 1])

        self.main = nn.Sequential(
            # Input: (batch, 1, 1, 1)

            # First Transpose Convolution
            nn.ConvTranspose2d(1, 64, kernel_size=4, stride=1, padding=0, bias=False),  # Output: (batch, 64, 4, 4)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Second Transpose Convolution
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # Output: (batch, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Third Transpose Convolution
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # Output: (batch, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Fourth Transpose Convolution
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  # Output: (batch, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Output Layer
            nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False),  # Output: (batch, 1, 32, 32)
            nn.Tanh()  # Outputs to (-1, 1)
        )

    def forward(self, input):
        return self.main(input)