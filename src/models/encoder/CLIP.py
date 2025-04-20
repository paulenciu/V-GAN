import clip
from src.models.encoder.AbstractEncoder import AbstractEncoder
from torch import nn


class CLIP(AbstractEncoder):

    def __init__(self):
        super(CLIP, self).__init__(has_decoder=False)
        self.encoder, _ = clip.load("ViT-B/32")
        self.encoder = CLIPEncoder(encoder=self.encoder)

    def get_encoder_and_freeze(self):
        for param in self.encoder.encoder.parameters():
            param.requires_grad = False
        return self.encoder

class CLIPEncoder(nn.Module):

    def __init__(self, encoder):
        super(CLIPEncoder, self).__init__()
        self.encoder = encoder
        self.encoder.eval()

    def forward(self, image):
        return self.encoder.encode_image(image)


