import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional
from src.models.generator.AbstractGenerator import AbstractGenerator


class SelfAttentionBlock(nn.Module):
    """
    Simple self-attention block.
    You can replace this with a more sophisticated version if desired.
    """

    def __init__(self, in_dim):
        super(SelfAttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # B: batch size, C: channels, H: height, W: width
        B, C, H, W = x.size()

        # query: (B, C/8, H*W)
        query = self.query_conv(x).view(B, -1, H * W)
        # key:   (B, C/8, H*W)
        key = self.key_conv(x).view(B, -1, H * W)
        # value: (B, C,   H*W)
        value = self.value_conv(x).view(B, -1, H * W)

        # attention: (B, H*W, H*W)
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention, dim=-1)

        # out: (B, C, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return x + self.gamma * out


class GeneratorConvLinearMappingBigSigmV3(AbstractGenerator):
    def __init__(self, latent_size: Optional[torch.Tensor] = None,
                 use_attention=True):
        super(GeneratorConvLinearMappingBigSigmV3, self).__init__()

        if latent_size is None:
            self._noise_dim = torch.tensor([64, 1, 1])
        else:
            self._noise_dim = latent_size

        latent_size = self._noise_dim[0].item()

        # -------------------------------
        #   ENCODER-ISH PATH
        # -------------------------------
        # Instead of a single 4x4 projection,
        # we do 8x8 + 16x16 expansions with skip layers.
        self.init = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> shape [512, 4, 4]

        self.down1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # -> [512, 8, 8]
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> shape [256, 8, 8]

        self.down2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # -> [256,16,16]
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> shape [128, 16, 16]

        # -------------------------------
        #   OPTIONAL ATTENTION
        # -------------------------------
        self.use_attention = use_attention
        if self.use_attention:
            self.attn = SelfAttentionBlock(128)

        # -------------------------------
        #   DECODER-ISH PATH
        # -------------------------------
        # We'll upsample back to 32x32, reusing the “skipped” features
        # from the earlier expansions for better detail.

        # Up from 16x16 -> 8x8
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),  # back to 8x8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Combine with down1 (which is at 8x8)
        self.combine1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Up from 8x8 -> 4x4
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Combine with init (which is at 4x4, 512 channels)
        self.combine2 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Finally, upsample from 4x4 to 32x32
        # (can do stepwise or do it in fewer steps)
        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()  # 1-channel mask, range [0,1]
        )

    def forward(self, z):
        # Encoder-ish forward
        x0 = self.init(z)  # -> [512, 4, 4]
        x1 = self.down1(x0)  # -> [256, 8, 8]
        x2 = self.down2(x1)  # -> [128, 16,16]

        # Optional self-attention
        if self.use_attention:
            x2 = self.attn(x2)

        # Decoder-ish forward
        # 1) 16x16 -> 8x8
        u1 = self.up1(x2)  # -> [256,8,8]
        c1 = torch.cat([u1, x1], dim=1)  # skip connection from x1
        c1 = self.combine1(c1)  # -> [256,8,8]

        # 2) 8x8 -> 4x4
        u2 = self.up2(c1)  # -> [512,4,4]
        c2 = torch.cat([u2, x0], dim=1)  # skip connection from x0
        c2 = self.combine2(c2)  # -> [512,4,4]

        # 3) 4x4 -> 32x32
        out = self.up_final(c2)  # -> [1, 32, 32]
        return out

    def sample_subspace_masks(self, noise):
        """
        Repeat across channels if needed.
        In your case, you might only need 1-channel if you’re just dealing with masks.
        """
        mask = self.forward(noise)
        return mask.repeat(1, 3, 1, 1)  # E.g., if your pipeline expects 3-ch version
