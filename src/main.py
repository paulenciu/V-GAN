import torch
import os

from pyod.models.lof import LOF

from src.data.dataset.MVTecADDataset import MVTecADDataset

from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4 import GeneratorOneChannelV4
from src.outlier_detection import launch_outlier_detection_experiments

from src.vmmdref.VMMDDiagonal1Channel import VMMDDiagonal1Channel
from torchvision.transforms import transforms

from src.vmmdref.VMMDDiagonal3Channel import VMMDDiagonal3Channel
from src.vmmdref.penalty.MMDLossPenalty import MMDLossL2Penalty, MMDLossPenaltyJoin, \
    MMDLossDiscretePenalty
from src.models.autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
import numpy as np
from sklearn.model_selection import ParameterSampler
from src.data.dataset_loader import load_data

if __name__ == '__main__':
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')
    torch.autograd.set_detect_anomaly(True)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((64, 64))
        ]
    )

    # cats_dataset = Cifar10CatsDataset("../datasets/cifar10")
    x_train, x_test, y_test = load_data(dataset_name="MVTec_AD", category=["bottle"], image_size=(224, 224))
    penalty_weight = 1
    lr = .1

    vmmd = VMMDDiagonal1Channel(
        filename=f"emb_bottle_mvtec_ad_1D",
        weight_decay=0.09999999999999999,  # TODO maybe smaller, 0?
        seed=333,
        momentum=0.8,
        lr=lr,
        epochs=2000,
        batch_size=256,
        path_to_directory="../experiments/remote",
        penalty=MMDLossPenaltyJoin(
            mmd_loss_1=MMDLossL2Penalty(1),
            mmd_loss_2=MMDLossDiscretePenalty(1),
            weight=1
        )
    )
    noise_dim = 256
    vmmd.fit(x_train, ResNet18AutoEncoder(), GeneratorOneChannelV4(noise_dim, x_train.image_shape))



