from pytorch_pretrained_vit import ViT
from torch import nn

from src.models.encoder.AbstractEncoder import AbstractEncoder


class ViTransformer(AbstractEncoder):

    def __init__(self, model="B_16_imagenet1k"):
        super(ViTransformer, self).__init__(has_decoder=False)
        self.encoder = ViT(model, pretrained=True, image_size=224)
        self.encoder.fc = nn.Identity()
        self.encoder.eval()

    def get_encoder_and_freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        return self.encoder