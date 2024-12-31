from torch import nn

from src.models.Generator import upper_softmax


class ResNetXDetachedDecoder(nn.Module):

    def __init__(self, original_generator):
        super(ResNetXDetachedDecoder, self).__init__()
        self.model = original_generator
        self.softmax = upper_softmax()
        self.latent_size = original_generator.latent_size

        # resetting the parameters
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x