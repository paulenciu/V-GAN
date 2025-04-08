import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np



class Cifar10AutoEncoder(nn.Module):

    def __init__(self):
        super(Cifar10AutoEncoder, self).__init__()
        self.model = AutoEncoder()
        autoencoder_model_pth = torch.load('../models/checkpoint_ep599_itir_999.pkl', map_location=torch.device('cpu'))
        self.model.load_state_dict(autoencoder_model_pth)

    def get_encoder(self):
        return self.model.encoder

    def get_encoder_input_shape(self):
        return torch.Size([3, 32, 32])

    def get_decoder_input_shape(self):
        return torch.Size([64, 56, 56])

    def get_encoder_and_freeze(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def get_decoder_and_freeze(self):
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        return self.model.decoder


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=16,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=32,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=64,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,  # input height
                out_channels=32,  # n_filters
                kernel_size=2,  # filter size
                stride=2,  # filter movement/step
                padding=0,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=32,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=2,  # filter size
                stride=2,  # filter movement/step
                padding=0,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=16,  # input height
                out_channels=3,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=3,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.ReLU(),  # activation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

