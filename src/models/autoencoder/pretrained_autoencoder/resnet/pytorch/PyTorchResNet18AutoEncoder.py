import torch

from src.utils.preprocessing import normalize_images


class PyTorchResNet18AutoEncoder:
    """
     RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
     The images have to be loaded in to a range of [0, 1] and then
     normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
     """
    def __init__(self):
        self.encoder = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.encoder.eval()

    def get_encoder_and_freeze(self):

        for param in self.encoder.parameters():
            param.requires_grad = False

        return self.encoder

