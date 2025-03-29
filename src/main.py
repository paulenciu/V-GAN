import torch
import os

from src.data.dataset_type import DatasetType
from src.models.autoencoder.pretrained_autoencoder.resnet.imagenet.ResNet50AutoEncoder import ResNet50AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.ResNetTrainer import ResNetTrainer
from src.data.dataset_loader import load_data

from src.models.autoencoder.pretrained_autoencoder.resnet.pytorch.PyTorchResNet18AutoEncoder import \
    PyTorchResNet18AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.pytorch.PyTorchResNet50AutoEncoder import \
    PyTorchResNet50AutoEncoder
from src.models.generator.diagonal_matrix.embedding.GeneratorRes50ConvV2 import GeneratorRes50ConvV2
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelSNGN import GeneratorOneChannelSNGN

from src.run.embeddingspace.config.EmbeddingVMMDBaseConfiguration import EmbeddingVMMDBaseConfiguration
from src.run.pipeline.VMMDPipeline import pretrained_vmmd_embedding_experiment, launch_vmmd_experiment, \
    launch_vmmd_embedding_config
from src.run.pixelspace.config.vmmd.VMMDBaseConfiguration import VMMDBaseConfiguration
from src.utils.preprocessing import normalize_images, normalize_features, min_max_scaling
from src.vmmd.MMDLossConstrained import MixtureRQLinear


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

    # dataset = load_data(dataset_type=DatasetType.MVTEC_AD, category="bottle", image_size=(224, 224), standardize=True, train=True)
    #
    # model = PyTorchResNet50AutoEncoder()
    # encoder = model.encoder
    # decoder = model.decoder
    #
    # trainer = ResNetTrainer(encoder, decoder, dataset)
    # trainer.train()

    #run_all_vmmd_od_benchmark()
    #launch_all_od_experiments()
    config = [
        VMMDBaseConfiguration(
            dataset_type=DatasetType.OCCCIFAR10,
            dateset_category="cat",
            n_subspace_sample=2,
            batch_size=250,
            add_to_title="ptres50_mk",
            epochs=2000,
            lr=0.0001,
            autoencoder=PyTorchResNet50AutoEncoder(),
            image_size_train=(224, 224),
            image_size_generator=(28, 28),
            image_size_od=(224, 224),
            standardize_data=True,
        ),
        VMMDBaseConfiguration(
            dataset_type=DatasetType.OCCFMNIST,
            dateset_category="Trouser",
            n_subspace_sample=2,
            batch_size=900,
            add_to_title="ptres18_mk",
            epochs=4000,
            lr=0.0001,
            autoencoder=PyTorchResNet18AutoEncoder(),
            image_size_train=(224, 224),
            image_size_generator=(28, 28),
            image_size_od=(224, 224),
            standardize_data=True,
        ),
        VMMDBaseConfiguration(
            dataset_type=DatasetType.OCCFMNIST,
            dateset_category="Trouser",
            n_subspace_sample=2,
            batch_size=250,
            add_to_title="ptres50_mk",
            epochs=4000,
            lr=0.001,
            autoencoder=PyTorchResNet50AutoEncoder(),
            image_size_train=(224, 224),
            image_size_generator=(28, 28),
            image_size_od=(224, 224),
            standardize_data=True,
        ),
        VMMDBaseConfiguration(
            dataset_type=DatasetType.OCCCIFAR10,
            dateset_category="cat",
            n_subspace_sample=2,
            batch_size=900,
            add_to_title="ptres18_mk",
            epochs=4000,
            lr=0.001,
            autoencoder=PyTorchResNet18AutoEncoder(),
            image_size_train=(224, 224),
            image_size_generator=(28, 28),
            image_size_od=(224, 224),
            standardize_data=True,
        ),
        VMMDBaseConfiguration(
            dataset_type=DatasetType.MVTEC_AD,
            dateset_category="bottle",
            n_subspace_sample=2,
            batch_size=1024,
            add_to_title="ptres50_mk",
            epochs=4000,
            lr=0.001,
            autoencoder=PyTorchResNet50AutoEncoder(),
            image_size_train=(224, 224),
            image_size_generator=(56, 56),
            image_size_od=(224, 224),
            standardize_data=True,
        ),
        VMMDBaseConfiguration(
            dataset_type=DatasetType.MVTEC_AD,
            dateset_category="bottle",
            n_subspace_sample=2,
            batch_size=1024,
            add_to_title="ptres50_mk",
            epochs=4000,
            lr=0.001,
            autoencoder=PyTorchResNet50AutoEncoder(),
            image_size_train=(224, 224),
            image_size_generator=(56, 56),
            image_size_od=(224, 224),
            standardize_data=True,
        ),
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
        #     batch_size=250,
        #     add_to_title="res50",
        #     epochs=3000,
        #     lr=0.0007,
        #     autoencoder=ResNet50AutoEncoder(),
        #     generator=GeneratorRes50ConvV2(512),
        #     kernel=MixtureRQLinear()
        # ),
    ]
    #pretrained_vmmd_embedding_experiment(config, "../experiments/remote/26-03/embedding_OCCCIFAR10[cat]_no_preprocessing__train224_lr=0.0005_bs=250_ep=2000_res50/models/generator_9.pt")
    launch_vmmd_experiment(config)
    #launch_vmmd_embedding_config(config)
    #pretrained_vmmd_embedding_experiment(config, "../experiments/remote/24-03/embedding_OCCCIFAR10[cat]_no_preprocessing__train224_lr=0.001_bs=800_ep=10000_res18_mk/models/generator_5.pt")
    # pretrained_vmmd_embedding_experiment(configs=config, path_to_pretrained_model="../experiments/remote/20-03/embedding_OCCFMNIST[Trouser]_no_preprocessing__train224_lr=0.001_bs=500_ep=1000_res18_fast/models/generator_9.pt")