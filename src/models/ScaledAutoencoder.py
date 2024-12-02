import torch
from torch import nn

from src.models.Autoencoder import ResNet50AutoEncoder, ResNet34AutoEncoder, ResNet18AutoEncoder


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
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)  # Keep 3 channels
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        # Pooling layers to downscale dimensions
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 224x224 -> 112x112
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 112x112 -> 56x56
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)  # 56x56 -> 28x28
        self.upsample_final = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=True)  # 28x28 -> 32x32

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)  # First downscale
        x = self.relu(self.conv2(x))
        x = self.pool2(x)  # Second downscale
        x = self.relu(self.conv3(x))
        x = self.pool3(x)  # Third downscale
        x = self.upsample_final(x)  # Resize to 32x32

        return x


class CombinedModel(nn.Module):
    def __init__(self, pretrained_model):
        super(CombinedModel, self).__init__()
        self.upscaler = CIFAR10Upscaler()

        self.pretrained_model = pretrained_model

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.downscaler = CIFAR10Downscaler()

    def forward(self, x):
        x = self.upscaler(x)
        x = self.pretrained_model(x)
        x = self.downscaler(x)
        return x


class ScaledAutoEncoder(nn.Module):
    def __init__(self, filepath):
        super(ScaledAutoEncoder, self).__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
        autoencoder_model_pth = torch.load(filepath)

        pretrained_model = None
        if "resnet50" in filepath:
            pretrained_model = ResNet50AutoEncoder()
        elif "resnet34" in filepath:
            pretrained_model = ResNet34AutoEncoder()
        elif "resnet18" in filepath:
            pretrained_model = ResNet18AutoEncoder()
        else:
            raise NotImplementedError("Model could not be loaded")

        model = CombinedModel(pretrained_model).to(device)
        model.load_state_dict(autoencoder_model_pth)
        self.upscaler = model.upscaler
        self.pretrained_model = model.pretrained_model
        self.downscaler = model.downscaler

    def forward(self, x):
        x = self.upscaler(x)
        x = self.pretrained_model(x)
        x = self.downscaler(x)
        return x

    def get_encoder(self):
        return ScaledEncoder(self)

    def get_decoder(self):
        return ScaledDecoder(self)

class ScaledEncoder(nn.Module):
    def __init__(self, model: ScaledAutoEncoder):
        super(ScaledEncoder, self).__init__()
        self.upscaler = model.upscaler
        self.encoder = model.pretrained_model.model.encoder

    def forward(self, x):
        x = self.upscaler(x)
        x = self.encoder(x)
        return x

class ScaledDecoder(nn.Module):
    def __init__(self, model: ScaledAutoEncoder):
        super(ScaledDecoder, self).__init__()
        self.downscaler = model.downscaler
        self.decoder = model.pretrained_model.model.decoder

    def forward(self, x):
        x = self.decoder(x)
        x = self.downscaler(x)
        return x
