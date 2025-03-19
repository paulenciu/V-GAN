import torch
import os

from pathlib import Path
from src.data.dataset_type import DatasetType
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

    config = [
        EmbeddingVMMDBaseConfiguration(
            dataset_type=DatasetType.MVTEC_AD,
            dateset_category="bottle",
            n_subspace_sample=100,
            batch_size=1024,
            add_to_title="res18",
            epochs=2,
            autoencoder=ResNet50AutoEncoder(),
        ),
    ]
    #
    # for mvtec_category in mvtec_categories:
    #     config.append(
    #         VMMDBaseConfiguration(
    #         dataset_type=DatasetType.MVTEC_AD,
    #         dateset_category=mvtec_category,
    #         image_size_od=(256,256),
    #         image_size_generator=(64, 64),
    #         image_size_train=(256,256),
    #         preprocessing_fn=normalize_features,
    #         standardize_data=False,
    #         n_subspace_sample=2,
    #     ))
    #
    # for fashionmnist_category in fashionmnist_categories:
    #     config.append(
    #         VMMDBaseConfiguration(
    #             dataset_type=DatasetType.OCCFMNIST,
    #             dateset_category=fashionmnist_category,
    #             image_size_od=(28,28),
    #             image_size_generator=(28,28),
    #             image_size_train=(28,28),
    #             preprocessing_fn=normalize_features,
    #             standardize_data=False,
    #             n_subspace_sample=2
    #         )
    #     )

    # for cifar10_category in cifar10_classes:
    #     config.append(
    #         VMMDBaseConfiguration(
    #             dataset_type=DatasetType.OCCCIFAR10,
    #             dateset_category=cifar10_category,
    #             image_size_od=(32,32),
    #             image_size_generator=(32,32),
    #             image_size_train=(32,32),
    #             preprocessing_fn=normalize_features,
    #             standardize_data=False,
    #             n_subspace_sample=2
    #         )
    #     )
    launch_vmmd_embedding_config(config)
    #pretrained_vmmd_experiment(config, "../experiments/remote/OCCFMNIST_normalize_features_train28_od28_lr=0.001_bs=1024_ep=2001_None/models/generator_9.pt")
