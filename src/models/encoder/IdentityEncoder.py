from torch import nn

from src.models.encoder.AbstractEncoder import AbstractEncoder


class IdentityEncoder(AbstractEncoder):

    def __init__(self):
        super(IdentityEncoder, self).__init__()

    def forward(self, x):
        return x