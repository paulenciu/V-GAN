import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Self-Attention layer as proposed in:
    "Self-Attention Generative Adversarial Networks" (Zhang et al.)
    """

    def __init__(self, in_channels):
        super().__init__()
        # The channel size for the latent representations f, g, h
        # typically is reduced for efficiency (in_channels // 8, etc.)
        self.in_channels = in_channels
        self.conv_f = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_g = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_h = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

        # Gamma is a learnable parameter that allows the network to
        # determine how much attention to use
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (batch_size, in_channels, height, width)
        """
        batch_size, C, H, W = x.size()

        # Compute f, g, and h
        f = self.conv_f(x)  # (batch_size, in_channels//8, H, W)
        g = self.conv_g(x)  # (batch_size, in_channels//8, H, W)
        h = self.conv_h(x)  # (batch_size, in_channels//2, H, W)

        # Reshape to (batch_size, -1, H*W)
        f_flat = f.view(batch_size, -1, H * W)  # (batch_size, in_channels//8, N)
        g_flat = g.view(batch_size, -1, H * W)  # (batch_size, in_channels//8, N)
        h_flat = h.view(batch_size, -1, H * W)  # (batch_size, in_channels//2, N)

        # Compute attention: beta = softmax(f^T * g)
        # f^T -> shape (batch_size, N, in_channels//8)
        # f^T * g -> shape (batch_size, N, N)
        beta = torch.bmm(f_flat.transpose(1, 2), g_flat)
        beta = self.softmax(beta)  # attention map

        # Self-attention output: o = h * beta
        # h_flat -> (batch_size, in_channels//2, N)
        # (h_flat * beta) -> (batch_size, in_channels//2, N)
        o = torch.bmm(h_flat, beta.transpose(1, 2))  # => (batch_size, in_channels//2, N)

        # Reshape o to match the original spatial dimensions
        o = o.view(batch_size, -1, H, W)  # => (batch_size, in_channels//2, H, W)
        o = self.conv_v(o)  # => (batch_size, in_channels, H, W)

        x_out = x + self.gamma * o
        return x_out
