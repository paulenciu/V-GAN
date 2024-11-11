import torch
import torch.nn as nn

class CIFAR10Upscaler(nn.Module):
    def __init__(self):
        super(CIFAR10Upscaler, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)  # Keep 3 channels
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 32x32 -> 64x64
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 64x64 -> 128x128
        self.upsample3 = nn.Upsample(scale_factor=1.75, mode='bilinear', align_corners=True)  # 128x128 -> 224x224
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.upsample1(x)
        x = self.relu(self.conv2(x))
        x = self.upsample2(x)
        x = self.relu(self.conv3(x))
        x = self.upsample3(x)
        return x

class CIFAR10Downscaler(nn.Module):
    def __init__(self):
        super(CIFAR10Downscaler, self).__init__()
        # Convolutional layers to downscale dimensions while keeping 3 channels
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        # Pooling layers to downscale spatial dimensions
        self.pool1 = nn.AdaptiveAvgPool2d((128, 128))  # 224x224 -> 128x128
        self.pool2 = nn.AdaptiveAvgPool2d((64, 64))  # 128x128 -> 64x64
        self.pool3 = nn.AdaptiveAvgPool2d((32, 32))  # 64x64 -> 32x32

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        return x
