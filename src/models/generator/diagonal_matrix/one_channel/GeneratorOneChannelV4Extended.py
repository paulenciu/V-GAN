from cmath import log

import torch
from torch import nn

from src.models.Generator import UpperSoftmax1D
from src.models.generator.AbstractGenerator import AbstractGenerator


class GeneratorOneChannelV4Extended(AbstractGenerator):

    def __init__(self, latent_size, image_shape, temperature=1.0, epsilon=1e-11):

        if isinstance(latent_size, torch.Tensor):
            latent_size = latent_size.item()

        super(GeneratorOneChannelV4Extended, self).__init__()

        self._noise_dim = torch.tensor([latent_size])
        self._img_shape = image_shape
        self.temperature = temperature
        self.epsilon = epsilon

        img_size = image_shape[2] * image_shape[2]
        rel_size = int(img_size/latent_size)
        self.latent_size = latent_size
        self.img_size = img_size
        amount_layers = 6
        self.increase = log(rel_size, amount_layers).real

        layers = [self.get_layer(layer) for layer in range(1, amount_layers)]
        layers += [self.get_layer(amount_layers, last=True)]
        self.layers = nn.Sequential(*layers)

        self.upper_softmax = UpperSoftmax1D()
        self.softmax = nn.Softmax(dim=-1)

    def get_layer(self, layer: int, last=False):
        input_size = round(pow(self.increase, layer - 1) * self.latent_size)
        output_size = round(pow(self.increase, layer) * self.latent_size)

        layer = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.8),
        )
        last_layer = nn.Sequential(
            nn.Linear(input_size, self.img_size),
        )

        self.softmax = nn.Softmax(dim=1)
        self.upper_softmax = UpperSoftmax1D()

        return last_layer if last else layer

    def forward(self, x, mode="train"):
        logits = self.layers(x)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + self.epsilon) + self.epsilon)
        y = (logits + gumbel_noise) / self.temperature
        if mode == "train":
            return self.softmax(y)

        return self.upper_softmax(y)

    def sample_subspace_masks(self, noise, mode="train"):
        activation = self.forward(noise, mode)
        activation_upscaled = activation.repeat(1, self._img_shape[0])
        if mode == "test":
            binary_activation = torch.greater_equal(activation_upscaled, 1 / activation.shape[1])
            activation_upscaled = binary_activation

        return activation_upscaled.view(-1, *self._img_shape)