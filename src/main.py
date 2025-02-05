import torch
import os
import pandas as pd
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR

from src.data.dataset.MVTecADDataset import MVTecADDataset
from src.data.dataset.SyntheticImageDataset import SyntheticImageDataset
from src.models.autoencoder.resnet.ResNet50AutoEncoder import ResNet50AutoEncoder

from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4 import GeneratorOneChannelV4
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV5 import GeneratorOneChannelV5
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV5Extended import GeneratorOneChannelV5Extended

from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV6 import GeneratorOneChannelV6
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV7 import GeneratorOneChannelV7
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV8 import GeneratorOneChannelV8

from src.outlier_detection import launch_outlier_detection_experiments
from src.vmmdref.VMMDConvLinearMappingRef import VMMDConvLinearMappingRef

from src.vmmdref.VMMDDiagonal1Channel import VMMDDiagonal1Channel
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from src.data.dataset_type import DatasetType

from src.vmmdref.VMMDDiagonal3Channel import VMMDDiagonal3Channel
from src.vmmdref.penalty.MMDLossPenalty import MMDLossL2Penalty, MMDLossPenaltyJoin, \
    MMDLossDiscretePenalty, MMDDiversityPenalty, MMDLossPenalty, MMDSparsityPenalty, KLDivergencePenalty, MMDGMMLoss, \
    MMDTVPenalty, MMDSigmoidPenalty, MMDSigmoidPenalty2
from src.models.autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
import numpy as np
from sklearn.model_selection import ParameterSampler
from src.data.dataset_loader import load_data

if __name__ == '__main__':
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()

    x_train_mvtec, _, _ = load_data(dataset_type=DatasetType.MVTEC_AD, category=["bottle"], image_size=(224, 224))
    #x_train_cifar, _, _ = load_data(dataset_type=DatasetType.CIFAR10, category=["3"], image_size=(224, 224))
    #x_train, _, _ = load_data(dataset_type=DatasetType.SYNTHETIC, category=["1"], image_size=(224, 224))
    #x_train = SyntheticImageDataset(5000, image_size=(64, 64), lower_half_white=False)
    #x_train_fmnist, _, _ = load_data(dataset_type=DatasetType.FASHION_MNIST, category=["1"], image_size=(224, 224))

    penalty_weight = 1
    lr = .1
    weight_decay = 0.01
    seed = 333
    momentum = 0.8
    epochs = 10000
    batch_size = 512

    # dict = launch_outlier_detection_experiments(
    #     path_to_generator="../experiments/remote/02-02/benchmark_sig_wo_emb_fmnist_1D_wd=0.01_s=333_m=0.8_bs=128/models/generator_5.pt",
    #     dataset_type=DatasetType.FASHION_MNIST,
    #     category=["1"],
    #     base_estimators=[LUNAR()],
    # )
    # df = pd.DataFrame([dict])
    # df.to_csv("../experiments/remote/02-02/benchmark_sig_wo_emb_fmnist_1D_wd=0.01_s=333_m=0.8_bs=128/ode.csv",
    #           index=False)
    #
    # dict = launch_outlier_detection_experiments(
    #     path_to_generator="../experiments/remote/02-02/benchmark_sig_wo_emb_cifar_1D_wd=0.01_s=333_m=0.8_bs=128/models/generator_4.pt",
    #     dataset_type=DatasetType.CIFAR10,
    #     category=["3"],
    #     base_estimators=[LUNAR()],
    # )
    # df = pd.DataFrame([dict])
    # df.to_csv("../experiments/remote/02-02/benchmark_sig_wo_emb_cifar_1D_wd=0.01_s=333_m=0.8_bs=128/ode.csv",
    #           index=False)
    #
    # dict = launch_outlier_detection_experiments(
    #     path_to_generator="../experiments/remote/02-02/benchmark_sig_w_emb_mvtec_1D_wd=0.01_s=333_m=0.8_bs=128/models/generator_11.pt",
    #     dataset_type=DatasetType.MVTEC_AD,
    #     category=["bottle"],
    #     base_estimators=[LUNAR()],
    # )
    # df = pd.DataFrame([dict])
    # df.to_csv("../experiments/remote/02-02/benchmark_sig_w_emb_mvtec_1D_wd=0.01_s=333_m=0.8_bs=128/ode.csv",
    #           index=False)

    mmd_loss_l2_weight = 2e-3

    vmmd = VMMDDiagonal1Channel(
        filename=f"224_sig_wo_emb_mvtec_1D_wd={weight_decay}_s={seed}_m={momentum}_bs={batch_size}",
        weight_decay=weight_decay,  # TODO maybe smaller, 0?
        seed=seed,
        momentum=momentum,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        path_to_directory="../experiments/remote",
        penalty=MMDLossPenaltyJoin(
            weight=penalty_weight,
            mmd_loss_1=MMDLossDiscretePenalty(1),
            mmd_loss_2=MMDLossL2Penalty(mmd_loss_l2_weight),
        ),
    )
    noise_dim = 256
    vmmd.fit(x_train_mvtec, ResNet50AutoEncoder(), GeneratorOneChannelV8(noise_dim, x_train_mvtec.image_shape))