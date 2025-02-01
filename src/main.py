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
from torchvision.transforms import transforms

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
    x_train, _, _ = load_data(dataset_name="CIFAR10", category=["3"], image_size=(64, 64))
    #x_train = SyntheticImageDataset(5000, image_size=(64, 64), lower_half_white=False)
    #x_train, _, _ = load_data(dataset_name="FASHION_MNIST", category=["1"], image_size=(224, 224))
    penalty_weight = 1
    lr = .1

    vmmd = VMMDDiagonal1Channel(
        filename=f"64_no_emb_r18_cifar10_1D-5",
        weight_decay=0.00099999999999,  # TODO maybe smaller, 0?
        seed=333,
        momentum=0.8,
        lr=lr,
        epochs=2000,
        batch_size=512,
        path_to_directory="../experiments/remote",
        penalty=MMDLossPenaltyJoin(
            weight=1,
            mmd_loss_1=MMDGMMLoss(5),
            mmd_loss_2=MMDSparsityPenalty(5),
        )
    )

    noise_dim = 256
    vmmd.fit(x_train, ResNet18AutoEncoder(), GeneratorOneChannelV6(noise_dim, x_train.image_shape))



