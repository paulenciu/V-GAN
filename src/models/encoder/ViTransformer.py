from pytorch_pretrained_vit import ViT
from torch import nn
class ViTransformer:

    def __init__(self, model="B_16_imagenet1k"):
        self.encoder = ViT(model, pretrained=True)
        self.encoder.fc = nn.Identity()
        self.encoder.eval()

    def get_encoder_and_freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        return self.encoder