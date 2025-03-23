import torch
import os

from pathlib import Path

from pyod.models.feature_bagging import FeatureBagging
from pyod.models.lunar import LUNAR

from src.data.dataset.occ.OCCFMNIST import OCCFMNIST
from src.data.dataset_type import DatasetType
from src.models.autoencoder.pretrained_autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.ResNet50AutoEncoder import ResNet50AutoEncoder
from src.models.generator.diagonal_matrix.embedding.GeneratorRes18 import GeneratorRes18
from src.models.generator.diagonal_matrix.embedding.GeneratorRes50 import GeneratorRes50
from src.models.generator.diagonal_matrix.embedding.GeneratorRes50Conv import GeneratorRes50Conv
from src.models.generator.diagonal_matrix.embedding.GeneratorRes50ConvV2 import GeneratorRes50ConvV2

from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelBatchDiscrimination import \
    GeneratorOneChannelBatchDiscrimination
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelSNBD import GeneratorOneChannelSNBD
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelSNGN import GeneratorOneChannelSNGN
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelSpectralNorm import \
    GeneratorOneChannelSpectralNorm
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4Softmax import GeneratorOneChannelV4Softmax
from src.run.embeddingspace.config.EmbeddingVMMDBaseConfiguration import EmbeddingVMMDBaseConfiguration
from src.run.pipeline.BaselinePipeline import launch_baseline_experiment
from src.run.pipeline.VMMDPipeline import pretrained_vmmd_experiment, launch_vmmd_experiment, \
    launch_vmmd_embedding_config, run_all_vmmd_od_benchmark, pretrained_vmmd_embedding_experiment, \
    launch_all_od_experiments
from src.run.pixelspace.config.vmmd.VMMDBaseConfiguration import VMMDBaseConfiguration
from src.run.pixelspace.config.vmmd.VMMDTestConfiguration import VMMDTestConfiguration
from src.utils.preprocessing import normalize_features, normalize_images
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannel import GeneratorOneChannel


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
    #run_all_vmmd_od_benchmark()
    #launch_all_od_experiments()
    config = [
        EmbeddingVMMDBaseConfiguration(
            dataset_type=DatasetType.MVTEC_AD,
            dateset_category="bottle",
            n_subspace_sample=2,
            batch_size=450,
            add_to_title="res50_fast",
            epochs=10000,
            lr=0.001,
            autoencoder=ResNet50AutoEncoder(),
            generator=GeneratorRes50ConvV2(512),
        ),
        EmbeddingVMMDBaseConfiguration(
            dataset_type=DatasetType.OCCCIFAR10,
            dateset_category="cat",
            n_subspace_sample=2,
            batch_size=450,
            add_to_title="res50_fast",
            epochs=2000,
            lr=0.0005,
            autoencoder=ResNet50AutoEncoder(),
            generator=GeneratorRes50ConvV2(512),
        ),
    ]
    launch_vmmd_embedding_config(config)
    # pretrained_vmmd_embedding_experiment(configs=config, path_to_pretrained_model="../experiments/remote/20-03/embedding_OCCFMNIST[Trouser]_no_preprocessing__train224_lr=0.001_bs=500_ep=1000_res18_fast/models/generator_9.pt")