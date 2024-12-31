import torch
from torch import nn


class GeneratorConvLinearMappingSmall(nn.Module):

    def __init__(self):
        super(GeneratorConvLinearMappingSmall, self).__init__()

        self.noise_dim = torch.tensor([3, 1, 1])


        self.main = nn.Sequential(
            nn.ConvTranspose2d(3, 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()  #outputs to (-1, 1)
        )

    def forward(self, input):
        return self.main(input)