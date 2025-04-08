import torch
import os

from src.data.dataset_type import DatasetType
from src.models.autoencoder.pretrained_autoencoder.cifar.CifarAutoEncoder import Cifar10AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.imagenet.ResNet50AutoEncoder import ResNet50AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.pytorch.PyTorchResNet18AutoEncoder import \
    PyTorchResNet18AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.pytorch.PyTorchResNet50AutoEncoder import \
    PyTorchResNet50AutoEncoder
from src.models.encoder.ViTransformer import ViTransformer
from src.models.generator.diagonal_matrix.embedding.GeneratorRes50ConvV2 import GeneratorRes50ConvV2
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelSNGN import GeneratorOneChannelSNGN

from src.run.embeddingspace.config.EmbeddingVMMDBaseConfiguration import EmbeddingVMMDBaseConfiguration
from src.run.pipeline.VMMDPipeline import pretrained_vmmd_embedding_experiment, launch_vmmd_experiment, \
    run_all_vmmd_od_benchmark, run_vmmd_od_benchmark
from src.run.pixelspace.config.vmmd.VMMDBaseConfiguration import VMMDBaseConfiguration
from src.utils.preprocessing import normalize_images


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
    #run_all_vmmd_od_benchmark()
    #run_all_vmmd_od_benchmark()
    #launch_all_od_experiments()
    config = [
        VMMDBaseConfiguration(
            dataset_type=DatasetType.OCCCIFAR10,
            dateset_category="cat",
            n_subspace_sample=100,
            batch_size=1024,
            add_to_title="l2loss",
            epochs=2000,
            lr=0.001,
        ),
        VMMDBaseConfiguration(
            dataset_type=DatasetType.OCCFMNIST,
            dateset_category="Trouser",
            n_subspace_sample=100,
            batch_size=1024,
            add_to_title="l2loss",
            epochs=2000,
            lr=0.001,
            image_size_train=(28, 28),
            image_size_od=(28, 28),
            image_size_generator=(28, 28),
        ),
        VMMDBaseConfiguration(
            dataset_type=DatasetType.MVTEC_AD,
            dateset_category="bottle",
            n_subspace_sample=100,
            batch_size=1024,
            add_to_title="l2loss",
            epochs=2000,
            lr=0.001,
            image_size_train=(256, 256),
            image_size_od=(256, 256),
            image_size_generator=(64, 64),
        ),

        # VMMDBaseConfiguration(
        #     dataset_type=DatasetType.OCCCIFAR10,
        #     dateset_category="cat",
        #     n_subspace_sample=100,
        #     batch_size=900,
        #     add_to_title="ptres50",
        #     epochs=2000,
        #     lr=0.001,
        #     autoencoder=PyTorchResNet50AutoEncoder(),
        #     image_size_train=(224, 224),
        #     image_size_generator=(28, 28),
        #     image_size_od=(224, 224),
        #     preprocessing_fn=normalize_images,
        #     standardize_data=True,
        # ),
        # EmbeddingVMMDBaseConfiguration(
        #     dataset_type=DatasetType.OCCCIFAR10,
        #     dateset_category="cat",
        #     n_subspace_sample=100   ,
        #     batch_size=450,
        #     add_to_title="res50_fast",
        #     epochs=10000,
        #     lr=0.001,
        #     autoencoder=ResNet50AutoEncoder(),
        #     generator=GeneratorRes50ConvV2(512),
        # ),
        # EmbeddingVMMDBaseConfiguration(
        #     dataset_type=DatasetType.OCCCIFAR10,
        #     dateset_category="cat",
        #     n_subspace_sample=2,
        #     batch_size=450,
        #     add_to_title="res50_fast",
        #     epochs=2000,
        #     lr=0.0005,
        #     autoencoder=ResNet50AutoEncoder(),
        #     generator=GeneratorRes50ConvV2(512),
        # ),
    ]
    launch_vmmd_experiment(config)
    #pretrained_vmmd_embedding_experiment(config, "../experiments/remote/24-03/embedding_OCCCIFAR10[cat]_no_preprocessing__train224_lr=0.001_bs=800_ep=10000_res18_mk/models/generator_5.pt")
    # pretrained_vmmd_embedding_experiment(configs=config, path_to_pretrained_model="../experiments/remote/20-03/embedding_OCCFMNIST[Trouser]_no_preprocessing__train224_lr=0.001_bs=500_ep=1000_res18_fast/models/generator_9.pt")