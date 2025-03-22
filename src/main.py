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
    launch_vmmd_embedding_config, run_all_vmmd_od_benchmark, pretrained_vmmd_embedding_experiment
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

    # config = [
    #     VMMDBaseConfiguration(
    #         dataset_type=DatasetType.OCCFMNIST,
    #         dateset_category="Trouser",
    #         n_subspace_sample=2,
    #         add_to_title="default",
    #         epochs=2000,
    #         lr=0.001,
    #         preprocessing_fn=normalize_images,
    #         generator=GeneratorOneChannel(latent_size=128, image_shape=(3, 32, 32)),
    #     ),
    #     VMMDBaseConfiguration(
    #         dataset_type=DatasetType.OCCFMNIST,
    #         dateset_category="Trouser",
    #         n_subspace_sample=2,
    #         add_to_title="bd",
    #         epochs=2000,
    #         lr=0.001,
    #         preprocessing_fn=normalize_images,
    #         generator=GeneratorOneChannelBatchDiscrimination(latent_size=128, image_shape=(3, 32, 32)),
    #     ),
    #     VMMDBaseConfiguration(
    #         dataset_type=DatasetType.OCCFMNIST,
    #         dateset_category="Trouser",
    #         n_subspace_sample=2,
    #         add_to_title="sn",
    #         epochs=2000,
    #         lr=0.001,
    #         preprocessing_fn=normalize_images,
    #         generator=GeneratorOneChannelSpectralNorm(latent_size=128, image_shape=(3, 32, 32)),
    #     ),
    #     VMMDBaseConfiguration(
    #         dataset_type=DatasetType.OCCFMNIST,
    #         dateset_category="Trouser",
    #         n_subspace_sample=2,
    #         add_to_title="sngn",
    #         epochs=2000,
    #         lr=0.001,
    #         preprocessing_fn=normalize_images,
    #         generator=GeneratorOneChannelSNGN(latent_size=128, image_shape=(3, 32, 32)),
    #     ),
    #     VMMDBaseConfiguration(
    #         dataset_type=DatasetType.OCCFMNIST,
    #         dateset_category="Trouser",
    #         n_subspace_sample=2,
    #         add_to_title="snbd",
    #         epochs=2000,
    #         lr=0.001,
    #         preprocessing_fn=normalize_images,
    #         generator=GeneratorOneChannelSNBD(latent_size=128, image_shape=(3, 32, 32)),
    #     ),
    #     VMMDBaseConfiguration(
    #         dataset_type=DatasetType.OCCFMNIST,
    #         dateset_category="Trouser",
    #         n_subspace_sample=2,
    #         add_to_title="all",
    #         epochs=2000,
    #         lr=0.001,
    #         preprocessing_fn=normalize_images,
    #         generator=GeneratorOneChannelV4Softmax(latent_size=128, image_shape=(3, 32, 32)),
    #     ),
    # ]
    #
    # launch_vmmd_experiment(config)

    config = [
        EmbeddingVMMDBaseConfiguration(
            dataset_type=DatasetType.OCCCIFAR10,
            dateset_category="cat",
            n_subspace_sample=2,
            batch_size=500,
            add_to_title="res18_fast",
            epochs=1000,
            lr=0.001,
            preprocessing_fn=normalize_features,
        ),
    ]

    pretrained_vmmd_embedding_experiment(configs=config, path_to_pretrained_model="../experiments/remote/21-03/embedding_OCCCIFAR10[cat]_no_preprocessing__train224_lr=0.0001_bs=500_ep=1000_res18_fast_gen50/models/generator_1.pt")