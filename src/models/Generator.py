import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F



# Regular function definition does not appear to work properly within a Sequential definition of a network in Pytorchs
class upper_softmax2D(nn.Module):
    def __init__(self):
        super().__init__()  # Dummy intialization as there is no parameter to learn

    def forward(self, x, d):
        x_flattened = x.view(x.size(0), -1)
        x_flattened = torch.nn.functional.softmax(x_flattened, 1)
        x_flattened = torch.less(x_flattened, 1/d)*x_flattened + \
            torch.greater_equal(x_flattened, 1/d)
        x = x_flattened.view(x.size(0), x.size(1), x.size(2), x.size(3))
        return x

class upper_softmax1D(nn.Module):
    def __init__(self):
        super().__init__()  # Dummy intialization as there is no parameter to learn

    def forward(self, x, d):
        x = torch.nn.functional.softmax(x, 1)
        x = torch.less(x, 1 / d) * x + \
            torch.greater_equal(x, 1 / d)
        return x

class upper_lower_softmax(nn.Module):
    def __init__(self):
        super().__init__()  # Dummy intialization as there is no parameter to learn

    def forward(self, x):
        x = torch.nn.functional.softmax(x, 1)
        selected = torch.greater_equal(x, 1/x.shape[1])
        x = x*selected + (~selected)*1e-08
        return x

