from torch import nn

from src.models.encoder.AbstractEncoder import AbstractEncoder


class IdentityAutoEncoder(AbstractEncoder):

    def __init__(self):
        super(IdentityAutoEncoder, self).__init__()

    def forward(self, x):
        return x

    def get_encoder_and_freeze(self):
        return self