import torch
from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty
from torch import nn


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2).to(device)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        n_samples = L2_distances.shape[0]
        self.bandwidth = L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        bandwidth = self.get_bandwidth(L2_distances)
        multipliers = self.bandwidth_multipliers
        return torch.exp(-L2_distances[None, ...] / (bandwidth * multipliers)[:, None, None]).sum(dim=0)


class MMDLossConstrained(nn.Module):
    '''
    Constrained loss by the number of features selected
    '''

    def __init__(self, kernel=RBF(), penalty=MMDLossNoPenalty()):
        super().__init__()
        self.kernel = kernel
        self.penalty = penalty
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    def forward(self, X, Y, U):
        K = self.kernel(torch.vstack([X, Y]))
        self.bandwidth = self.kernel.bandwidth
        self.bandwidth_multipliers = self.kernel.bandwidth_multipliers
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:]
        YY = K[X_size:, X_size:].mean()

        print("Y:", Y[0])
        print("X:", X[0])
        print("XX: ", XX)
        print("YY: ", YY)

        mmd_loss = XX - 2 * XY + YY

        total_loss = mmd_loss + self.penalty.get_weighted_penalty(U)

        return total_loss, mmd_loss

class MMDLossSquareRootConstrained(nn.Module):
    '''
    Constrained loss by the number of features selected
    '''

    def __init__(self, kernel=RBF(), penalty=MMDLossNoPenalty()):
        super().__init__()
        self.kernel = kernel
        self.penalty = penalty
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    def forward(self, X, Y, U):
        K = self.kernel(torch.vstack([X, Y]))
        self.bandwidth = self.kernel.bandwidth
        self.bandwidth_multipliers = self.kernel.bandwidth_multipliers
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()

        mmd_loss = XX - 2 * XY + YY
        mmd_loss_square_root = torch.sqrt(mmd_loss)
        total_loss = mmd_loss_square_root + self.penalty.get_weighted_penalty(U)

        return total_loss, mmd_loss_square_root