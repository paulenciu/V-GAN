from src.models.encoder.pretrained_autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder


class AutoEncoderManager:

    def __init__(self):
        resnet18 = ResNet18AutoEncoder()

        self.autoencoders = {
            resnet18.__class__.__name__: resnet18
        }

    def get_autoencoder(self, name):
        return self.autoencoders[name]