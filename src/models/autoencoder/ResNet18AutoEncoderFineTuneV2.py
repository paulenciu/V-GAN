from src.models.autoencoder.pretrained_autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
from torch import nn


class ResNet18AutoEncoderFineTuneV2(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained autoencoder and freeze it
        self.resnet18 = ResNet18AutoEncoder()
        self.resnet18.eval()

        # Get encoder and decoder
        self.pretrained_encoder = self.resnet18.get_encoder()
        self.pretrained_decoder = self.resnet18.get_decoder()

        # Freeze pretrained parameters
        for param in self.pretrained_encoder.parameters():
            param.requires_grad_(False)
        for param in self.pretrained_decoder.parameters():
            param.requires_grad_(False)

        self.encoder_last_layer = self._find_ith_conv_layer(self.pretrained_encoder, i=-1)
        self.decoder_last_layer = self._find_ith_conv_layer(self.pretrained_decoder, i=0)

        # Unfreeze parameters in the last layers
        for param in self.encoder_last_layer.parameters():
            param.requires_grad_(True)
        for param in self.decoder_last_layer.parameters():
            param.requires_grad_(True)

    def _find_ith_conv_layer(self, module, i):
        conv_layers = []
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                conv_layers.append(m)
        if not conv_layers:
            raise ValueError("No convolutional layers found in module")
        return conv_layers[i]

    def forward(self, x):
        x_encoded = self.pretrained_encoder(x)
        x_reconstructed = self.pretrained_decoder(x_encoded)
        return x_encoded.flatten(1), x_reconstructed

    def encode(self, x):
        return self.pretrained_encoder(x).flatten(1)

    def freeze_encoder(self):
        """Freeze the last layer of the encoder"""
        for param in self.encoder_last_layer.parameters():
            param.requires_grad_(False)

    def freeze_decoder(self):
        """Freeze the last layer of the decoder"""
        for param in self.decoder_last_layer.parameters():
            param.requires_grad_(False)

    def unfreeze_encoder(self):
        """Unfreeze the last layer of the encoder"""
        for param in self.encoder_last_layer.parameters():
            param.requires_grad_(True)

    def unfreeze_decoder(self):
        """Unfreeze the last layer of the decoder"""
        for param in self.decoder_last_layer.parameters():
            param.requires_grad_(True)

    def freeze_detector(self):
        """Freeze both encoder and decoder last layers"""
        self.freeze_encoder()
        self.freeze_decoder()

    def get_trainable_parameters(self):
        """Return parameters from the last layers of encoder and decoder"""
        return list(self.encoder_last_layer.parameters()) + list(self.decoder_last_layer.parameters())