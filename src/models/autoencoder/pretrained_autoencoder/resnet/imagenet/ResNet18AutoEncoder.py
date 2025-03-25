import torch

from src.models.encoder.AbstractEncoder import AbstractEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.imagenet.ResNetConfig import get_configs
from src.models.autoencoder.pretrained_autoencoder.resnet.imagenet.RestNetAutoEncoder import ResNetAutoEncoder


class ResNet18AutoEncoder(AbstractEncoder):
    def __init__(self):
        super(ResNet18AutoEncoder, self).__init__()
        autoencoder_model_pth = torch.load('../models/caltech256-resnet18.pth', map_location=torch.device('cpu'))
        config, bottleneck = get_configs('resnet18')
        self.model = ResNetAutoEncoder(config, bottleneck)

        state_dict = autoencoder_model_pth['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k  # remove "module." prefix
            new_state_dict[name] = v

        # Load the modified state_dict
        self.model.load_state_dict(new_state_dict)

    def forward(self, x):
        x_encoded = self.model.encoder(x)
        x_reconstructed = self.model.decoder(x_encoded)
        return x_encoded, x_reconstructed

    def get_encoder(self):
        return self.model.encoder

    def get_encoder_input_shape(self):
        return torch.Size([3, 224, 224])

    def get_decoder_input_shape(self):
        return torch.Size([512, 7, 7])

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

