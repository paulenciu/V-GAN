import torch
from torch import nn

from ResidualBlock import ResidualBlock


class GeneratorConvLinearMappingResidual(nn.Module):
    """
    The generator model that transforms a noise tensor into a mask tensor.
    """
    def __init__(self):
        super(GeneratorConvLinearMappingResidual, self).__init__()

        self.noise_dim = torch.tensor([3, 3, 3])

        # Fully connected layer to expand the input noise
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 3, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        # Upsampling layers with residual blocks
        self.main = nn.Sequential(
            # Reshape to (batch_size, 512, 4, 4)
            ResidualBlock(512),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),  # (8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # (16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),   # (32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),              # Final output
            nn.Tanh()  # Outputs values between -1 and 1
        )

    def forward(self, input):
        # Flatten the input noise tensor
        x = input.view(input.size(0), -1)
        # Expand and reshape
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        # Generate the output mask
        x = self.main(x)
        return x
