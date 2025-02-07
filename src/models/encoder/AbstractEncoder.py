from abc import ABC

from torch import nn


class AbstractEncoder(ABC, nn.Module):

    def __init__(self):
        super().__init__()

