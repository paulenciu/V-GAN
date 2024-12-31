import torch
from torch import nn

from src.models.autoencoder.resnet.ResNetConfig import get_configs
from src.models.autoencoder.resnet.RestNetAutoEncoder import ResNetAutoEncoder


class ResNet18AutoEncoder(nn.Module):
    def __init__(self):
        super(ResNet18AutoEncoder, self).__init__()
        autoencoder_model_pth = torch.load('../../models/caltech256-resnet18.pth', map_location=torch.device('cpu'))
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
        x = self.model(x)
        return x


    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        self.model.decoder.latent_size = 512
        return self.model.decoder

