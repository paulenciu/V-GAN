import torch

from src.dataset.Cifar10CatsDataset import Cifar10CatsDataset
from src.dataset.FashionMNISTTrousersDataset import FashionMNISTTrousersDataset
from src.models.generator.convolution.GeneratorConvLinearMappingBigSigmV2 import GeneratorConvLinearMappingBigSigmV2
from src.models.generator.convolution.GeneratorConvLinearMappingBigSigmV3 import GeneratorConvLinearMappingBigSoftmax

from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV3 import GeneratorOneChannelV3
from src.models.generator.diagonal_matrix.three_channels.GeneratorThreeChannel import GeneratorThreeChannel
from src.vmmdref.VMMDConvLinearMappingRef import VMMDConvLinearMappingRef

from src.vmmdref.VMMDDiagonal1Channel import VMMDDiagonal1Channel
from torchvision.transforms import transforms
import os

from src.vmmdref.VMMDDiagonal3Channel import VMMDDiagonal3Channel
from src.vmmdref.penalty.MMDLossPenalty import MMDLossL2Penalty, MMDLossPenaltyJoin, \
    MMDLossDiscretePenalty
from src.models.autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
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
    torch.autograd.set_detect_anomaly(True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    cats_dataset = Cifar10CatsDataset("../datasets/cifar10")
    penalty_weight = 1
    lr = .1

    vmmd = VMMDConvLinearMappingRef(
        filename=f"conv_sm_w={penalty_weight}_lr={lr}_1",
        weight_decay=0.09999999999999999,
        seed=333,
        momentum=0.8,
        lr=lr,
        epochs=100,
        batch_size=256,
        path_to_directory="../experiments/local",
        penalty=MMDLossPenaltyJoin(
            mmd_loss_1=MMDLossDiscretePenalty(1),
            mmd_loss_2=MMDLossL2Penalty(5e-3),
            weight=penalty_weight),
    )
    noise_dim = 100
    vmmd.fit(cats_dataset, ResNet18AutoEncoder(), GeneratorConvLinearMappingBigSigmV2(torch.tensor([noise_dim, 1, 1])))

