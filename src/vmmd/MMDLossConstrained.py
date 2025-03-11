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


class RationalQuadratic(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Rational Quadratic Kernel.

        Args:
            alpha (float): Shape parameter. Controls the tail behavior of the kernel.
                           Higher values make the kernel behave more like the RBF kernel.
        """
        super().__init__()
        self.alpha = alpha

    def get_bandwidth(self, L2_distances):
        n_samples = L2_distances.shape[0]
        self.bandwidth = L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

    def forward(self, X):
        """
        Compute the Rational Quadratic Kernel matrix.

        Args:
            X (torch.Tensor): Input tensor of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Kernel matrix of shape (n_samples, n_samples).
        """
        self.bandwidth = self.get_bandwidth(X)
        L2_distances = torch.cdist(X, X) ** 2
        return (1 + L2_distances / (2 * self.alpha)) ** (-self.alpha)


class MixtureRQLinear(nn.Module):

    def get_bandwidth(self, L2_distances):
        n_samples = L2_distances.shape[0]
        self.bandwidth = L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

    def __init__(self, alphas=[0.2, 0.5, 1.0, 2.0, 5], linear_weight=1.0):
        """
        Mixture of Rational Quadratic Kernels with a Linear Kernel.

        Args:
            alphas (list of float): List of alpha values for the RQ kernels.
                                   Each alpha corresponds to a different RQ kernel.
            linear_weight (float): Weight for the linear kernel.
        """
        super().__init__()
        self.alphas = alphas
        self.linear_weight = linear_weight

    def forward(self, X):
        """
        Compute the mixture of RQ kernels and linear kernel.

        Args:
            X (torch.Tensor): Input tensor of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Combined kernel matrix of shape (n_samples, n_samples).
        """
        L2_distances = torch.cdist(X, X) ** 2
        self.bandwidth = self.get_bandwidth(L2_distances)

        rq_kernels = []
        for alpha in self.alphas:
            rq_kernel = (1 + L2_distances / (2 * alpha)) ** (-alpha)
            rq_kernels.append(rq_kernel)

        rq_mixture = torch.sum(torch.stack(rq_kernels), dim=0)

        linear_kernel = torch.matmul(X, X.T)

        combined_kernel = rq_mixture + self.linear_weight * linear_kernel

        return combined_kernel

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
        stack = torch.vstack([X, Y])
        K = self.kernel(stack)
        self.bandwidth = self.kernel.bandwidth
        #self.bandwidth_multipliers = self.kernel.bandwidth_multipliers
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()

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