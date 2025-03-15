import torch
import os

from pyod.models.feature_bagging import FeatureBagging
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR
from src.data.dataset.occ.OCCCifar10 import OCCCifar10
from src.models.autoencoder.ResNet18AutoEncoderFineTuneV2 import ResNet18AutoEncoderFineTuneV2
from src.models.autoencoder.pretrained_autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
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
            od_model=CombinedOutlierDetector(
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

        experiement.fit_pretrained_model(path_to_pretrained_model)
        experiement.evaluate_interval(ensemble_weight_start=0, ensemble_weight_end=1, step=1.0 / 10.0)

def launch_baseline_experiment(configs, od_model):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))
        baseline_experiments = OutlierDetectionBaselineExperiment(
            dataset_type=config.dataset_type,
            category=config.dateset_category,
            image_size_od=config.image_size_od,
            standardize_data=config.standardize_data,
            preprocessing_fn=config.preprocessing_fn,
            od_model=od_model,
        )

        baseline_experiments.fit()
        baseline_experiments.evaluate()

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

    vmmd_config = [
        VMMDBaseConfiguration(
            dataset_type=DatasetType.OCCCIFAR10,
            dateset_category="cat",
            encoder=ResNet18AutoEncoder().get_encoder_and_freeze(),
            add_to_title="resnet18",
            image_size_generator=(28, 28),
            image_size_od=(28, 28),
            image_size_train=(28, 28),
            lr=0.0005,
        ),
        VMMDBaseConfiguration(
            dataset_type=DatasetType.OCCFMNIST,
            dateset_category="Trouser",
            encoder=ResNet18AutoEncoder().get_encoder_and_freeze(),
            add_to_title="resnet18",
            image_size_generator=(28, 28),
            image_size_od=(28, 28),
            image_size_train=(28, 28),
            lr=0.0005,
            preprocessing_fn=normalize_images
        ),
        VMMDBaseConfiguration(
            dataset_type=DatasetType.OCCFMNIST,
            dateset_category="Trouser",
            encoder=ResNet18AutoEncoder().get_encoder_and_freeze(),
            add_to_title="resnet18",
            image_size_generator=(28, 28),
            image_size_od=(28, 28),
            image_size_train=(28, 28),
            lr=0.0005,
        )
    ]

    launch_vmmd_experiment(vmmd_config)

    config = [
        VGANBaseConfiguration(
            dataset_type=DatasetType.MVTEC_AD,
            dateset_category="bottle",
            epochs=10000,
            iternum_g=2000,
            iternum_d=1000,
            lr_g=0.00002,
            lr_d=0.00002,
            detector=ResNet18AutoEncoderFineTuneV2(),
        ),
        VGANBaseConfiguration(
            dataset_type=DatasetType.OCCCIFAR10,
            dateset_category="cat",
            epochs=10000,
            iternum_g=2000,
            iternum_d=1000,
            lr_g=0.00002,
            lr_d=0.00002,
            detector=ResNet18AutoEncoderFineTuneV2(),
        )
    ]

    launch_vgan_experiment(config)
