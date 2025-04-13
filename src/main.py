import torch
import os

from pyod.models.knn import KNN
from pyod.models.lof import LOF

from doc.table_generator import generate_latex_tables
from src.data.dataset_type import DatasetType
from src.models.autoencoder.pretrained_autoencoder.cifar.CifarAutoEncoder import Cifar10AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.imagenet.ResNet18AutoEncoder import ResNet18AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.imagenet.ResNet50AutoEncoder import ResNet50AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.pytorch.PyTorchResNet18AutoEncoder import \
    PyTorchResNet18AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.pytorch.PyTorchResNet50AutoEncoder import \
    PyTorchResNet50AutoEncoder
from src.models.encoder.ViTransformer import ViTransformer
from src.models.generator.diagonal_matrix.embedding.GeneratorRes18 import GeneratorRes18
from src.models.generator.diagonal_matrix.embedding.GeneratorRes50Conv import GeneratorRes50Conv
from src.models.generator.diagonal_matrix.embedding.GeneratorRes50ConvV2 import GeneratorRes50ConvV2
from src.models.generator.diagonal_matrix.embedding.GeneratorRes50V2 import GeneratorRes50V2
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelSNGN import GeneratorOneChannelSNGN

from src.run.embeddingspace.config.EmbeddingVMMDBaseConfiguration import EmbeddingVMMDBaseConfiguration
from src.run.pipeline.BaselinePipeline import launch_baseline_experiment, launch_all_baseline_experiments_fs, \
    mvtec_categories, cifar10_classes, fashionmnist_categories
from src.run.pipeline.VMMDPipeline import pretrained_vmmd_embedding_experiment, launch_vmmd_experiment, \
    run_all_vmmd_od_benchmark, run_vmmd_od_benchmark, launch_vmmd_embedding_config, pretrained_vmmd_experiment, \
    launch_vmmd_embedding_space_config
from src.run.pixelspace.config.vmmd.VMMDBaseConfiguration import VMMDBaseConfiguration
from src.run.pixelspace.config.vmmd.VMMDTestConfiguration import VMMDTestConfiguration
from src.utils.preprocessing import normalize_images
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

    import random

    random_seed = random.randint(0, 2 ** 32 - 1)

    for mv_class in mvtec_categories:
        torch.cuda.empty_cache()

        config = [
            EmbeddingVMMDBaseConfiguration(
                dataset_type=DatasetType.MVTEC_AD,
                dateset_category=mv_class,
                lr=1e-5,
                autoencoder=ResNet50AutoEncoder(),
                kernel=MixtureRQLinear(),
                add_to_title="r50",
                latent_size=1024,
                epochs=10000,
                generator=GeneratorRes50V2(latent_size=1024, image_shape=None),
                weight_decay=0.0
            )
        ]
        launch_vmmd_embedding_space_config(config)

