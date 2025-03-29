import torch
from torch import nn
from src.utils.preprocessing import normalize_images
from src.models.autoencoder.pretrained_autoencoder.resnet.pytorch.PyTorchResNet50Decoder import PyTorchResNet50Decoder


class PyTorchResNet50AutoEncoder:
    """
     RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
     The images have to be loaded in to a range of [0, 1] and then
     normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
     """
    def __init__(self, eval=True):
        self.encoder = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        self.decoder = PyTorchResNet50Decoder()

        if eval:
            self.encoder.eval()


    def get_encoder_and_freeze(self):

        for param in self.encoder.parameters():
            param.requires_grad = False

        return self.encoder

