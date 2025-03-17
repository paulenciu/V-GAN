import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.functional import tanh, sigmoid


class DyT(nn.Module):

    def __init__(self, alpha_init=1, gamma_init=0.5, beta_init=0.5, delta_init=0.5):
        super(DyT, self).__init__()
        self.alpha = Parameter(torch.ones(1) * alpha_init)
        self.gamma = Parameter(torch.ones(1) * gamma_init)
        self.beta = Parameter(torch.ones(1) * beta_init)
        self.delta = Parameter(torch.ones(1) * delta_init)

    def forward(self, x):
        x = tanh(self.alpha * x + self.delta)
        return self.gamma * x + self.beta