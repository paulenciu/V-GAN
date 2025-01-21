import torch
import torchvision
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelResidualBigInv import \
    GeneratorOneChannelResidualBigInv
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV2 import GeneratorOneChannelV2
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV3 import GeneratorOneChannelV3
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4 import GeneratorOneChannelV4
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV5 import GeneratorOneChannelV5

from src.vmmdref.VMMDDiagonal1Channel import VMMDDiagonal1Channel
from torch import nn
from torchvision.transforms import transforms
import gc
import os
from dataset.SyntheticImageDataset import SyntheticImageDataset
from src.vmmdref.penalty.MMDLossPenalty import MMDLossL2Penalty, MMDLossNoPenalty, MMDLossPenaltyJoin, \
    MMDLossDiscretePenalty, MMDLossDiscreteExponentialPenalty, MMDLossDiscreteJenkeJenkePenalty
from src.models.autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
from src.models.autoencoder.resnet.ResNet50AutoEncoder import ResNet50AutoEncoder
from src.models.generator.convolution.GeneratorConvLinearMappingBigSigmV3 import GeneratorConvLinearMappingBigSoftmax
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
        'batch_size':    [64, 128, 256, 512],
        'epochs':        [1000, 2000, 3000, 4000],
        'lr':            [0.5, 0.1] ,    # e.g., [1e-4, 1e-3, 1e-2, 1e-1]
        'momentum':      np.linspace(0.8, 0.99, 5),  # e.g., [0.8, 0.85, 0.9, 0.95, 0.99]
        'seed':          [777, 111, 222, 333],
        'weight_decay':  np.logspace(-4, -1, 4), # e.g., [1e-4, 1e-3, 1e-2, 1e-1]
        'latent_size': [16, 32, 64, 128]
    }

    # Randomly sample n_iter configurations from the above distributions
    sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state)

    # Convert the iterator to a list of dictionaries
    param_list = list(sampler)
    return param_list



if __name__ == '__main__':
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')

    # transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         #       transforms.Normalize(mean=[0.4955, 0.4564, 0.4155], std=[0.2568, 0.2523, 0.2580])
    #     ]
    # )
    # dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
    #                                        transform=transform)
    # cats_dataset = [(img, label) for (img, label) in dataset if label == 3]
    s_dataset = SyntheticImageDataset(10000)
    hyperparameter_list = generate_hyperparams(n_iter=10, random_state=888)
    torch.autograd.set_detect_anomaly(True)

    penalty_weight = 1
    lr = .5

    vmmd = VMMDDiagonal1Channel(
        filename=f"test_syn_1_jp_tr=sigm,test=sigm_one_channel_w={penalty_weight}_lr={lr}_4",
        weight_decay=0.09999999999999999,
        seed=333,
        momentum=0.8,
        lr=lr,
        epochs=200,
        batch_size=256,
        path_to_directory="/home/i40/enciup/V-GAN/experiments/remote",
        penalty=MMDLossPenaltyJoin(
            mmd_loss_1=MMDLossDiscretePenalty(1),
            mmd_loss_2=MMDLossL2Penalty(1e-7),
            weight=penalty_weight,
            )
    )
    noise_dim = 100
    vmmd.fit(s_dataset, ResNet18AutoEncoder(), GeneratorOneChannelV5(noise_dim))

