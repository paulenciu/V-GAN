from pl_bolts.models.autoencoders import VAE
from torch import nn

class VAutoencoder(nn.Module):

    def __init__(self):
        super(VAutoencoder, self).__init__()
        self.model = VAE(input_height=32)
        self.model = self.model.from_pretrained('cifar10-resnet18')
        self.model.freeze()

    def forward(self, input):
        return self.model.forward(input)

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        self.model.decoder.latent_size = 512 #FIXME what does this mean
        return self.model.decoder