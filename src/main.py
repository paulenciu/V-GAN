import torch
import torchvision
from src.vmmdref.VMMDDiagonal1Channel import VMMDDiagonal1Channel
from torch import nn
from torchvision.transforms import transforms
import gc

from src.models.autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
from src.models.autoencoder.resnet.ResNet50AutoEncoder import ResNet50AutoEncoder
from src.models.generator.convolution.GeneratorConvLinearMappingBigSigmV2 import GeneratorConvLinearMappingBigSigmV2
from src.vmmdref.VMMDConvLinearMappingRef import VMMDConvLinearMappingRef
import numpy as np
from sklearn.model_selection import ParameterSampler

def generate_hyperparams(n_iter=10, random_state=777):
    """
    Generates a list of dictionaries, each containing a random combination
    of hyperparameters for model tuning.

    :param n_iter:        How many random hyperparameter sets to generate.
    :param random_state:  Seed for reproducibility.
    :return:              A list of dictionaries with hyperparameter configurations.
    """

    # Define your hyperparameter search space
    param_distributions = {
        'batch_size':    [128, 256, 500, 1024],
        'epochs':        [1000, 500, 700, 400],
        'lr':            [0.5, 0.1, 0.05, 0.01, 0.005] ,    # e.g., [1e-4, 1e-3, 1e-2, 1e-1]
        'momentum':      np.linspace(0.8, 0.99, 5),  # e.g., [0.8, 0.85, 0.9, 0.95, 0.99]
        'seed':          [777, 111, 222, 333],
        'weight_decay':  np.logspace(-4, -1, 4), # e.g., [1e-4, 1e-3, 1e-2, 1e-1]
        'latent_size': [64, 256, 512, 1024]
    }

    # Randomly sample n_iter configurations from the above distributions
    sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state)

    # Convert the iterator to a list of dictionaries
    param_list = list(sampler)
    return param_list

from src.models.generator.diagonal_matrix.three_channels.GeneratorThreeChannel import GeneratorThreeChannel
from src.vmmdref.VMMDDiagonal3Channel import VMMDDiagonal3Channel




if __name__ == '__main__':
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            #       transforms.Normalize(mean=[0.4955, 0.4564, 0.4155], std=[0.2568, 0.2523, 0.2580])
        ]
    )
    dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                           transform=transform)
    cats_dataset = [(img, label) for (img, label) in dataset if label == 3]

    hyperparameter_list = generate_hyperparams(n_iter=10, random_state=777)
    for i in range(5, len(hyperparameter_list)):
        vmmd = VMMDDiagonal3Channel(
            filename=f"three_channel_hypparam_test_0_{i}",
            weight_decay=hyperparameter_list[i]['weight_decay'],
            seed=hyperparameter_list[i]['seed'],
            momentum=hyperparameter_list[i]['momentum'],
            lr=0.5,
            epochs=2,
            batch_size=hyperparameter_list[i]['batch_size'],
            path_to_directory="/Users/paulenciu/PycharmProjects/V-GAN/experiments/local")
        noise_dim = hyperparameter_list[i]['latent_size']
        vmmd.fit(cats_dataset, ResNet18AutoEncoder(), GeneratorThreeChannel(noise_dim))