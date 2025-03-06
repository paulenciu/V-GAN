import torch
from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty
from torch import nn


class MMDLossConstrainedFixKernel(nn.Module):
    def __init__(self, penalty=MMDLossNoPenalty(), bandwidths=[1, 2, 4, 8, 16]):
        super().__init__()
        self.penalty = penalty
        self.bandwidths = bandwidths

    def forward(self, x, y, u):
        mmd_loss = 0.0
        for bandwidth in self.bandwidths:
            xx = torch.cdist(x, x)
            yy = torch.cdist(y, y)
            xy = torch.cdist(x, y)

            k_xx = torch.exp(-xx / (2 * bandwidth**2))
            k_yy = torch.exp(-yy / (2 * bandwidth**2))
            k_xy = torch.exp(-xy / (2 * bandwidth**2))

            mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
            mmd_loss += mmd

        mmd_loss /= len(self.bandwidths)
        return mmd_loss, mmd_loss