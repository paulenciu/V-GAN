import torch
from torch import nn

from src.models.autoencoder.pretrained_autoencoder.resnet.imagenet.ResNet18AutoEncoder import ResNet18AutoEncoder


class ResNet18AutoEncoderFineTuning(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained autoencoder and freeze it
        self.resnet18 = ResNet18AutoEncoder()
        self.resnet18.eval()

        # Get encoder and decoder
        self.pretrained_encoder = self.resnet18.get_encoder_and_freeze()
        self.pretrained_decoder = self.resnet18.get_decoder()

        # Freeze pretrained parameters
        for param in self.pretrained_encoder.parameters():
            param.requires_grad_(False)
        for param in self.pretrained_decoder.parameters():
            param.requires_grad_(False)


        self.pretrained_latent_shape = torch.Size([512, 7, 7])
        self.pretrained_latent_dim_flattened = 512 * 7 * 7
        self.finetune_latent_dim = int(self.pretrained_latent_dim_flattened / 4)

        self.encoder_last_layer = nn.Sequential(
            nn.Linear(self.pretrained_latent_dim_flattened, self.finetune_latent_dim),
            nn.BatchNorm1d(self.finetune_latent_dim),
            nn.ReLU(),
        )
        self.decoder_last_layer = nn.Sequential(
            nn.Linear(self.finetune_latent_dim, self.pretrained_latent_dim_flattened),
            nn.BatchNorm1d(self.pretrained_latent_dim_flattened),
            nn.ReLU(),
        )

    def forward(self, x):
        batch_size = x.size(0)

        x_encoded = self.pretrained_encoder(x)
        x_flat = x_encoded.flatten(1)
        x_encoded_flat = self.encoder_last_layer(x_flat)

        x_decoder_flat = self.decoder_last_layer(x_encoded_flat)
        x_decoder_input = x_decoder_flat.view(batch_size, *self.pretrained_latent_shape)
        x_reconstructed = self.pretrained_decoder(x_decoder_input)

        return x_encoded_flat, x_reconstructed

    def encode(self, x):
        x_encoded = self.pretrained_encoder(x)
        x_flat = x_encoded.flatten(1)
        x_encoded_flat = self.encoder_last_layer(x_flat)
        return x_encoded_flat

    def freeze_encoder(self):
        for p in self.encoder_last_layer.parameters():
            p.requires_grad = False

    def freeze_decoder(self):
        for p in self.decoder_last_layer.parameters():
            p.requires_grad = False

    def unfreeze_decoder(self):
        for p in self.decoder_last_layer.parameters():
            p.requires_grad = True

    def unfreeze_encoder(self):
        for p in self.encoder_last_layer.parameters():
            p.requires_grad = True

    def unfreeze_detector(self):
        self.unfreeze_decoder()
        self.unfreeze_encoder()

    def freeze_detector(self):
        self.freeze_decoder()
        self.freeze_encoder()

    def get_trainable_parameters(self):
        return list(self.encoder_last_layer.parameters()) + list(self.decoder_last_layer.parameters())