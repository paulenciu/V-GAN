import torch


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.shadow[name].mul_(self.decay).add_(param, alpha=1 - self.decay)
                param.copy_(self.shadow[name])  # Update model parameters with EMA
