from abc import ABC

from torch import nn


class AbstractEncoder(ABC, nn.Module):

    def __init__(self, has_decoder=False):
        super().__init__()
        self.has_decoder = has_decoder


