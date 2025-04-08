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
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelSNGN import GeneratorOneChannelSNGN

from src.run.embeddingspace.config.EmbeddingVMMDBaseConfiguration import EmbeddingVMMDBaseConfiguration
from src.run.pipeline.BaselinePipeline import launch_baseline_experiment, launch_all_baseline_experiments_fs, \
    mvtec_categories
from src.run.pipeline.VMMDPipeline import pretrained_vmmd_embedding_experiment, launch_vmmd_experiment, \
    run_all_vmmd_od_benchmark, run_vmmd_od_benchmark, launch_vmmd_embedding_config, pretrained_vmmd_experiment, \
    launch_vmmd_embedding_space_config
from src.run.pixelspace.config.vmmd.VMMDBaseConfiguration import VMMDBaseConfiguration
from src.run.pixelspace.config.vmmd.VMMDTestConfiguration import VMMDTestConfiguration
from src.utils.preprocessing import normalize_images
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

    #generate_latex_tables(metric="auc")
    #generate_latex_tables(metric="prauc")
    #launch_all_baseline_experiments_fs(KNN())
    #launch_baseline_experiment()
    #run_all_vmmd_od_benchmark()
    #run_all_vmmd_od_benchmark()
    #launch_all_od_experiments()
    # config = [
    #     VMMDBaseConfiguration(
    #     dataset_type=DatasetType.OCCFMNIST,
    #     dateset_category="T-shirt/top",
    #     image_size_od=(28, 28),
    #     preprocessing_fn=normalize_images,
    #     standardize_data=False,
    #     n_subspace_sample=100,
    #     ens_base_estimator=KNN()
    # )]

    import random

    random_seed = random.randint(0, 2 ** 32 - 1)

    for mv_class in mvtec_categories:
        config_1 = [
            EmbeddingVMMDBaseConfiguration(
                dataset_type=DatasetType.MVTEC_AD,
                dateset_category=mv_class,
                n_subspace_sample=100,
                batch_size=750,
                add_to_title="vit_emb",
                epochs=3000,
                lr=0.00001,
                seed=random_seed,
                autoencoder=ViTransformer(),
                generator=GeneratorRes18(64, 768),
                kernel=MixtureRQLinear(),
            ),
            EmbeddingVMMDBaseConfiguration(
                dataset_type=DatasetType.MVTEC_AD,
                dateset_category=mv_class,
                n_subspace_sample=100,
                batch_size=750,
                add_to_title="vit_emb",
                epochs=3000,
                lr=0.00001,
                seed=random_seed,
                autoencoder=ViTransformer(),
                generator=GeneratorRes18(64, 768),
                kernel=MixtureRQLinear(),
                weight_decay=0,
            ),
        ]
        launch_vmmd_embedding_space_config(configs=config_1)
        break
    #pretrained_vmmd_experiment(config, "../experiments/remote/21-03/OCCFMNIST[T-shirt-top]/models/generator_1.pt")
    #launch_vmmd_experiment(config)
    #launch_vmmd_embedding_config(config)
    #pretrained_vmmd_embedding_experiment(config, "../experiments/remote/24-03/embedding_OCCCIFAR10[cat]_no_preprocessing__train224_lr=0.001_bs=800_ep=10000_res18_mk/models/generator_5.pt")
    # pretrained_vmmd_embedding_experiment(configs=config, path_to_pretrained_model="../experiments/remote/20-03/embedding_OCCFMNIST[Trouser]_no_preprocessing__train224_lr=0.001_bs=500_ep=1000_res18_fast/models/generator_9.pt")