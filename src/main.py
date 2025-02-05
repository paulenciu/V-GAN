import pandas as pd
import torch
import os

from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR


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

    dict = launch_outlier_detection_experiments(
        path_to_generator="../experiments/remote/02-02/benchmark_sig_wo_emb_fmnist_1D_wd=0.01_s=333_m=0.8_bs=128/models/generator_5.pt",
        dataset_type=DatasetType.FASHION_MNIST,
        category=["1"],
        base_estimators=[LUNAR()],
    )
    df = pd.DataFrame([dict])
    df.to_csv("../experiments/remote/02-02/benchmark_sig_wo_emb_fmnist_1D_wd=0.01_s=333_m=0.8_bs=128/ode.csv", index=False)
