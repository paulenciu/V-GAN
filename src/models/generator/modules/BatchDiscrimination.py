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