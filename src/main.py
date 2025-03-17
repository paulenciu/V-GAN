from pathlib import Path

import torch
import os

from pyod.models.feature_bagging import FeatureBagging
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR
from src.data.dataset.occ.OCCCifar10 import OCCCifar10
from src.models.autoencoder.ResNet18AutoEncoderFineTuneV2 import ResNet18AutoEncoderFineTuneV2
from src.models.autoencoder.pretrained_autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
from src.models.autoencoder.pretrained_autoencoder.resnet.ResNet50AutoEncoder import ResNet50AutoEncoder
from src.od.CombinedOutlierDetectionV2 import CombinedOutlierDetectorV2
from src.run.OutlierDetectionBaselineExperiment import OutlierDetectionBaselineExperiment
from src.run.OutlierDetectionExperiment import OutlierDetectionExperiment
from src.od.CombinedOutlierDetector import CombinedOutlierDetector
from src.run.config.vgan.VGANBaseConfiguration import VGANBaseConfiguration
from src.run.config.vgan.VGANTestConfiguration import VGANTestConfiguration
from src.run.config.vmmd.VMMDBaseConfiguration import VMMDBaseConfiguration
from src.run.config.vmmd.VMMDTestConfiguration import VMMDTestConfiguration
from src.utils.preprocessing import normalize_features, normalize_images_col_softmax, normalize_images
from src.vgan.VGAN import VGAN
from src.vmmd.model.VMMDDiagonal1Channel import VMMDDiagonal1Channel

from src.data.dataset_type import DatasetType

fashionmnist_categories = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

mvtec_categories = [
    # "bottle",
    # "cable",
    # "capsule",
    # "carpet",
    # "grid",
    # "hazelnut",
    # "leather",
    # "metal_nut",
    # "pill",
    # "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper"
]

cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

def launch_vgan_experiment(configs):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))

        vmmd = VGAN(
            epochs=config.epochs, seed=config.seed, path_to_directory=config.path_to_directory,
            lr_G=config.lr_g, lr_D=config.lr_d, penalty=config.penalty, filename=config.filename,
            batch_size=config.batch_size, momentum=config.momentum, weight_decay=config.weight_decay,
            detector=config.detector, generator=config.generator, iternum_g=config.iternum_g, iternum_d=config.iternum_d,
        )

        experiement = OutlierDetectionExperiment(
            vmmd=vmmd,
            od_model=CombinedOutlierDetector(
                base_estimators=[config.ens_base_estimator],
                vmmd=vmmd, max_n_jobs=-1,
                preprocessing_fn=config.preprocessing_fn
            ),
            dataset_type=config.dataset_type,
            category=config.dateset_category,
            image_size_train=config.image_size_train,
            image_size_od=config.image_size_od,
            standardize_data=config.standardize_data,
            preprocessing_fn=config.preprocessing_fn,
            n_subspaces_sample=config.n_subspace_sample
        )

        experiement.fit()
        experiement.evaluate_interval(ensemble_weight_start=0, ensemble_weight_end=1, step=1.0 / 10.0)


def launch_vmmd_experiment(configs):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))

        vmmd = VMMDDiagonal1Channel(
            epochs=config.epochs, seed=config.seed, path_to_directory=config.path_to_directory,
            lr=config.lr, penalty=config.penalty, filename=config.filename,
            batch_size=config.batch_size, momentum=config.momentum, weight_decay=config.weight_decay,
            encoder=config.encoder, generator=config.generator
        )

        experiement = OutlierDetectionExperiment(
            vmmd=vmmd,
            od_model=CombinedOutlierDetectorV2(
                base_estimators=[config.ens_base_estimator],
                vmmd=vmmd, max_n_jobs=1,
                preprocessing_fn=config.preprocessing_fn
            ),
            dataset_type=config.dataset_type,
            category=config.dateset_category,
            image_size_train=config.image_size_train,
            image_size_od=config.image_size_od,
            standardize_data=config.standardize_data,
            preprocessing_fn=config.preprocessing_fn,
            n_subspaces_sample=config.n_subspace_sample
        )

        experiement.fit()
        experiement.evaluate_interval(ensemble_weight_start=0, ensemble_weight_end=1, step=1.0 / 10.0)

def pretrained_vmmd_experiment(configs, path_to_pretrained_model):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))

        vmmd = VMMDDiagonal1Channel(
            epochs=config.epochs, seed=config.seed, path_to_directory=config.path_to_directory,
            lr=config.lr, penalty=config.penalty, filename=config.filename,
            batch_size=config.batch_size, momentum=config.momentum, weight_decay=config.weight_decay,
            encoder=config.encoder, generator=config.generator
        )

        experiement = OutlierDetectionExperiment(
            vmmd=vmmd,
            od_model=CombinedOutlierDetector(
                base_estimators=[config.ens_base_estimator],
                vmmd=vmmd, max_n_jobs=-1,
                preprocessing_fn=config.preprocessing_fn
            ),
            dataset_type=config.dataset_type,
            category=config.dateset_category,
            image_size_train=config.image_size_train,
            image_size_od=config.image_size_od,
            standardize_data=config.standardize_data,
            preprocessing_fn=config.preprocessing_fn,
            n_subspaces_sample=config.n_subspace_sample
        )

        experiement.fit_pretrained_model(path_to_pretrained_model)
        experiement.evaluate_interval(ensemble_weight_start=0, ensemble_weight_end=1, step=1.0 / 10.0)

def launch_baseline_experiment(configs):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))
        baseline_experiments = OutlierDetectionBaselineExperiment(
            dataset_type=config.dataset_type,
            category=config.dateset_category,
            image_size_od=config.image_size_od,
            standardize_data=config.standardize_data,
            preprocessing_fn=config.preprocessing_fn,
            od_model=config.ens_base_estimator,
        )

        baseline_experiments.fit()
        baseline_experiments.evaluate()

def rerun_od_experiments():
    root_dir = Path("../experiments/remote/12-03/")
    for mvtec_category in mvtec_categories:
        prefix = str(DatasetType.MVTEC_AD.name) + "[" + str(mvtec_category) + "]"

        for dir in root_dir.iterdir():
            if dir.stem.startswith(prefix):

                config = [VMMDBaseConfiguration(
                    dataset_type=DatasetType.MVTEC_AD,
                    dateset_category=mvtec_category,
                    image_size_od=(256,256),
                    preprocessing_fn=normalize_images,
                    standardize_data=False,
                    n_subspace_sample=100
                )]

                path_to_generator =  str(dir / "models" / "generator_1.pt")

                pretrained_vmmd_experiment(config, path_to_generator)

    for fashionmnist_category in fashionmnist_categories:
        prefix = str(DatasetType.OCCFMNIST.name) + "[" + str(fashionmnist_category) + "]"
        for dir in root_dir.iterdir():
            if dir.stem.startswith(prefix):
                config = [VMMDBaseConfiguration(
                    dataset_type=DatasetType.OCCFMNIST,
                    dateset_category=fashionmnist_category,
                    image_size_od=(28,28),
                    preprocessing_fn=normalize_images,
                    standardize_data=False,
                    n_subspace_sample=100
                )]

                path_to_generator = str(dir / "models" / "generator_1.pt")

                pretrained_vmmd_experiment(config, path_to_generator)

    for cifar_category in cifar10_classes:
        prefix = str(DatasetType.CIFAR10.name) + "[" + str(cifar_category) + "]"
        for dir in root_dir.iterdir():
            if dir.stem.startswith(prefix):
                config = [VMMDBaseConfiguration(
                    dataset_type=DatasetType.CIFAR10,
                    dateset_category=cifar_category,
                    image_size_od=(32,32),
                    preprocessing_fn=normalize_images,
                    standardize_data=False,
                    n_subspace_sample=100
                )]

                path_to_generator = str(dir / "models" / "generator_1.pt")
                pretrained_vmmd_experiment(config, path_to_generator)


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
    rerun_od_experiments()
    # configs = [
    #     VMMDBaseConfiguration(
    #         dataset_type=DatasetType.OCCFMNIST,
    #         dateset_category="Trouser",
    #         image_size_train=(28, 28),
    #         image_size_generator=(28, 28),
    #         image_size_od=(28, 28),
    #         preprocessing_fn=normalize_images,
    #         n_subspace_sample=500
    #     ),
    # ]
    #launch_vmmd_experiment(configs)
    #pretrained_vmmd_experiment(configs, "../experiments/remote/OCCFMNIST_normalize_features_train28_od28_lr=0.001_bs=1024_ep=2001_None/models/generator_9.pt")
