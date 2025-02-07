from torch import nn
#from transformers import AutoModel


class NatAutoEncoder(nn.Module):
    pass
    # def __init__(self):
    #     super(NatAutoEncoder, self).__init__()
    #     self.model = AutoModel.from_pretrained("nateraw/pretrained_autoencoder-cifar10")
    #
    # def forward(self, x):
    #     x = self.model(x)
    #     return x
    #
    #
    # def get_encoder(self):
    #     return self.model.encoder
    #
    # def get_decoder(self):
    #     self.model.decoder.latent_size = 512 #FIXME what does this mean
    #     return self.model.decoder