import torch
import os

from pyod.models.feature_bagging import FeatureBagging
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR

from doc.table_generator import generate_pixel_space_tables, generate_encoded_space_tables, \
    generate_attention_table_on_baseline, generate_pixel_space_with_distance_table, generate_conover_iman_table
from src.data.dataset_type import DatasetType
from src.models.autoencoder.pretrained_autoencoder.cifar.CifarAutoEncoder import Cifar10AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.imagenet.ResNet18AutoEncoder import ResNet18AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.imagenet.ResNet50AutoEncoder import ResNet50AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.pytorch.PyTorchResNet18AutoEncoder import \
    PyTorchResNet18AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.pytorch.PyTorchResNet50AutoEncoder import \
    PyTorchResNet50AutoEncoder
from src.models.encoder.ViTransformer import ViTransformer
from src.models.generator.diagonal_matrix.embedding.GeneratorBig import GeneratorBig
from src.models.generator.diagonal_matrix.embedding.GeneratorRes18 import GeneratorRes18
from src.models.generator.diagonal_matrix.embedding.GeneratorRes50Conv import GeneratorRes50Conv
from src.models.generator.diagonal_matrix.embedding.GeneratorRes50ConvV2 import GeneratorRes50ConvV2
from src.models.generator.diagonal_matrix.embedding.GeneratorRes50V2 import GeneratorRes50V2
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelSNGN import GeneratorOneChannelSNGN
from src.run.embeddingspace.config.EmbeddingBaselineConfiguration import EmbeddingBaselineConfiguration

from src.run.embeddingspace.config.EmbeddingVMMDBaseConfiguration import EmbeddingVMMDBaseConfiguration
from src.run.pipeline.BaselinePipeline import launch_baseline_experiment, launch_all_baseline_experiments_fs, \
    mvtec_categories, cifar10_classes, fashionmnist_categories, launch_attention_baseline, \
    launch_baseline_on_embedding_space
from src.run.pipeline.VMMDPipeline import pretrained_vmmd_embedding_experiment, launch_vmmd_experiment, \
    run_all_vmmd_od_benchmark, run_vmmd_od_benchmark, launch_vmmd_embedding_config, pretrained_vmmd_experiment, \
    launch_vmmd_embedding_space_config, launch_pval_calculation, launch_all_od_experiments, rerun_od_experiments, \
    rerun_od_distance_experiments, pretrained_vmmd_distance_experiment
from src.run.pixelspace.config.vmmd.VMMDBaseConfiguration import VMMDBaseConfiguration
from src.run.pixelspace.config.vmmd.VMMDTestConfiguration import VMMDTestConfiguration
from src.utils.Plotter import plot_ens_dis_comparison
from src.utils.preprocessing import normalize_images, normalize_features
from src.vmmd.MMDLossConstrained import MixtureRQLinear, RBF
from src.vmmd.penalty.MMDLossPenalty import MMDLossL2Penalty


def configure_environment():
    """Set environment variables and PyTorch options."""
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    configure_environment()
    generate_conover_iman_table()