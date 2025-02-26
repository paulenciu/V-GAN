import torch.nn as nn

from src.models.autoencoder.pretrained_autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder

class ResNet18AutoEncoderFineTuning(nn.Module):

    def __init__(self):

        self.resnet18 = ResNet18AutoEncder()
        self.resnet18.eval() #Freeze autoencoder
        self.pretrained_encoder = self.resnet18.get_encoder()
        self.pretrained_decoder = self.resnet18.get_decoder()

        self.encoder_last_layer = nn.Sequential([
            nn.Linear(100, 100),
        ])

        self.decoder_last_layer = nn.Sequential([
            nn.Linear(100, 100),
        ])

    def forward(self, x):
        x = self.pretrained_encoder(x)
        x = x.flatten()
        x_encoded = self.encoder_last_layer(x)
        x = self.decoder_last_layer(x_encoded)
        x_reconstructed = self.pretrained_decoder(x)
        return x_encoded, x_reconstructed

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder