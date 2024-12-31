from torch import nn


class ResidualBlock(nn.Module):
    """
    A residual block that adds the input to the output of a sequence of layers.
    This helps in training deeper networks by mitigating the vanishing gradient problem.
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.main(x))