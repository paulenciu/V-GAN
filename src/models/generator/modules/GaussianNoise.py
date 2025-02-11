import numpy as np
import torch
from torch import nn


class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.1):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            shape = x.shape
            # Generate a NumPy array with the same shape, filled with random values in [0, 1) from gaus distribution
            noise = np.random.rand(*shape)
            noise = torch.from_numpy(noise).to(torch.float32).to(device=x.device).float() * self.stddev
            return x + noise
        return x