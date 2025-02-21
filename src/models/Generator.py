import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F



# Regular function definition does not appear to work properly within a Sequential definition of a network in Pytorchs
class UpperSoftmax2D(nn.Module):

    def __init__(self):
        super().__init__()  # Dummy intialization as there is no parameter to learn

    def stable_log_softmax(self, logits):
        logits_max = torch.max(logits, dim=-1, keepdim=True).values
        exps = torch.exp(logits - logits_max)
        return logits - logits_max - torch.log(torch.sum(exps, dim=-1, keepdim=True))

    def stable_softmax(self, x, dim=1):
        # Subtract the maximum value for numerical stability
        x_max = torch.max(x, dim=dim, keepdim=True).values
        x_stable = x - x_max
        # Compute the softmax
        return F.softmax(x_stable, dim=dim)

    def forward(self, x):
        x_flattened = x.view(x.size(0), -1)

        x_flattened = self.stable_softmax(x_flattened)

        # Apply thresholding
        threshold = 1 / x_flattened.shape[1]
        x_flattened = torch.where(x_flattened >= threshold, torch.tensor(1.0, device=x.device),
                                  torch.tensor(0.0, device=x.device))

        if torch.all(x_flattened):
            print(x)
            print(x_flattened)

        x = x_flattened.view(x.size(0), x.size(1), x.size(2), x.size(3))
        return x

class UpperSoftmax1D(nn.Module):

    def __init__(self):
        super().__init__()  # Dummy intialization as there is no parameter to learn

    def forward(self, x):
        d = x.shape[1]
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
