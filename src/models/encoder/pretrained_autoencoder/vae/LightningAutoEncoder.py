#from pl_bolts.models.autoencoders import AE
from torch import nn

class LightningAutoEncoder(nn.Module):
    pass
    # def __init__(self):
    #     super(LightningAutoEncoder, self).__init__()
    #     self.model = AE(input_height=32)
    #     self.model = self.model.from_pretrained('cifar10-resnet18')
    #     self.model.freeze()
    #
    # def forward(self, input):
    #     return self.model.forward(input)
    #
    # def get_encoder(self):
    #     return self.model.encoder