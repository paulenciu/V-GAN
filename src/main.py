import torch
import os

from pathlib import Path
from src.data.dataset_type import DatasetType
from src.models.autoencoder.pretrained_autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.ResNet50AutoEncoder import ResNet50AutoEncoder
from src.run.embeddingspace.config.EmbeddingVMMDBaseConfiguration import EmbeddingVMMDBaseConfiguration
from src.run.pipeline.VMMDPipeline import pretrained_vmmd_experiment, launch_vmmd_experiment, \
    launch_vmmd_embedding_config
from src.run.pixelspace.config.vmmd.VMMDBaseConfiguration import VMMDBaseConfiguration
from src.run.pixelspace.config.vmmd.VMMDTestConfiguration import VMMDTestConfiguration
from src.utils.preprocessing import normalize_features


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

    # config = [
    #     EmbeddingVMMDBaseConfiguration(
    #         dataset_type=DatasetType.OCCCIFAR10,
    #         dateset_category="cat",
    #         n_subspace_sample=5,
    #         batch_size=250,
    #         add_to_title="res50",
    #         epochs=50,
    #         autoencoder=ResNet50AutoEncoder(),
    #     ),
    # ]
    #
    # launch_vmmd_embedding_config(config)
    torch.cuda.empty_cache()

    config = [
        EmbeddingVMMDBaseConfiguration(
            dataset_type=DatasetType.OCCFMNIST,
            dateset_category="Trouser",
            n_subspace_sample=5,
            batch_size=250,
            add_to_title="res50",
            epochs=50,
            autoencoder=ResNet50AutoEncoder(),
        ),
    ]

    launch_vmmd_embedding_config(config)

