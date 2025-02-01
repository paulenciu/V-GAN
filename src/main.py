import torch
import os

from pyod.models.lof import LOF

from src.data.dataset.MVTecADDataset import MVTecADDataset
from src.data.dataset.SyntheticImageDataset import SyntheticImageDataset
from src.models.autoencoder.resnet.ResNet50AutoEncoder import ResNet50AutoEncoder

from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4 import GeneratorOneChannelV4
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV5 import GeneratorOneChannelV5
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV6 import GeneratorOneChannelV6

from src.outlier_detection import launch_outlier_detection_experiments
from src.vmmdref.VMMDConvLinearMappingRef import VMMDConvLinearMappingRef

from src.vmmdref.VMMDDiagonal1Channel import VMMDDiagonal1Channel
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from src.data.dataset_type import DatasetType

from src.vmmdref.VMMDDiagonal3Channel import VMMDDiagonal3Channel
from src.vmmdref.penalty.MMDLossPenalty import MMDLossL2Penalty, MMDLossPenaltyJoin, \
    MMDLossDiscretePenalty, MMDDiversityPenalty, MMDLossPenalty, MMDSparsityPenalty, KLDivergencePenalty, MMDGMMLoss, \
    MMDTVPenalty
from src.models.autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
import numpy as np
from sklearn.model_selection import ParameterSampler
from src.data.dataset_loader import load_data

if __name__ == '__main__':
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()

    #x_train, _, _ = load_data(dataset_name="MVTec_AD", category=["bottle"], image_size=(224, 224))
    #x_train, _, _ = load_data(dataset_type=DatasetType.CIFAR10, category=["3"], image_size=(64, 64))
    #x_train, _, _ = load_data(dataset_type=DatasetType.SYNTHETIC, category=["1", "2"], image_size=(64, 64))
    #x_train = SyntheticImageDataset(5000, image_size=(64, 64), lower_half_white=False)
    x_train, _, _ = load_data(dataset_type=DatasetType.FASHION_MNIST, category=["1"], image_size=(224, 224))

    penalty_weight = 1
    lr = .01
    weight_decay = 0.01
    seed = 333
    momentum = 0.8
    epochs = 4000
    batch_size = 512

    vmmd = VMMDDiagonal1Channel(
        filename=f"benchmark_usm_no_emb_syn_1D_wd={weight_decay}_s={seed}_m={momentum}_bs={batch_size}",
        weight_decay=weight_decay,  # TODO maybe smaller, 0?
        seed=seed,
        momentum=momentum,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        path_to_directory="../experiments/remote",
        penalty=MMDDiversityPenalty(1),
    )
    noise_dim = 256
    vmmd.fit(x_train, ResNet18AutoEncoder(), GeneratorOneChannelV4(noise_dim, x_train.image_shape))



