import torch
import os

from pyod.models.feature_bagging import FeatureBagging
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR
from src.data.dataset.occ.OCCCifar10 import OCCCifar10
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
    configs = [
        # VGANBaseConfiguration(
        #     dataset_type=DatasetType.OCCCIFAR10,
        #     dateset_category="cat",
        #     detector=ResNet18AutoEncoderFineTuneV2(),
        #     iternum_g=10,
        #     iternum_d=5,
        #     epochs=200,
        #     lr_g=0.0001,
        #     lr_d=0.0001,
        # ),
        # VMMDTestConfiguration(
        #     dataset_type=DatasetType.MVTEC_AD,
        #     dateset_category=["bottle"],
        #     epochs=2000,
        # ),
        VMMDBaseConfiguration(
            dataset_type=DatasetType.MVTEC_AD,
            dateset_category=["bottle"],
            image_size_generator=(64, 64),
            image_size_train=(256, 256),
            image_size_od=(256, 256),
            preprocessing_fn=normalize_images,
            n_subspace_sample=20,
            standardize_data=True,
        ),
        VMMDBaseConfiguration(
            dataset_type=DatasetType.MVTEC_AD,
            dateset_category=["bottle"],
            image_size_generator=(64, 64),
            image_size_train=(900, 900),
            image_size_od=(900, 900),
            preprocessing_fn=normalize_features,
            n_subspace_sample=20
        ),
        VMMDBaseConfiguration(
            dataset_type=DatasetType.MVTEC_AD,
            dateset_category=["bottle"],
            image_size_generator=(64, 64),
            image_size_train=(256, 256),
            image_size_od=(512, 512),
            preprocessing_fn=normalize_images,
        ),

        VMMDBaseConfiguration(
            dataset_type=DatasetType.MVTEC_AD,
            dateset_category=["bottle"],
            image_size_generator=(64, 64),
            image_size_train=(224, 224),
            image_size_od=(224, 224),
            epochs=2000,
            preprocessing_fn=normalize_images,
        ),
        VMMDBaseConfiguration(
            dataset_type=DatasetType.MVTEC_AD,
            dateset_category=["Trouser"],
            image_size_generator=(64, 64),
            image_size_train=(224, 224),
            image_size_od=(224, 224),
            epochs=2000,
            preprocessing_fn=normalize_features,
        ),
    ]

    baseline_experiments = OutlierDetectionBaselineExperiment(
        dataset_type=DatasetType.OCCCIFAR10,
        category="cat",
        image_size_od=(32, 32),
        standardize_data=False,
        preprocessing_fn=normalize_images,
        od_model=FeatureBagging(n_estimators=100, base_estimator=LOF())
    )

    baseline_experiments.fit()
    baseline_experiments.evaluate()

    for i, config in enumerate(configs):

        print("RUNNING EXPERIMENT", i, " FROM", len(configs))

        vmmd = VMMDDiagonal1Channel(
            epochs=config.epochs, seed=config.seed, path_to_directory=config.path_to_directory,
            lr=config.lr, penalty=config.penalty, filename=config.filename,
            batch_size=config.batch_size, momentum=config.momentum, weight_decay=config.weight_decay,
            encoder=config.encoder, generator=config.generator
        )

        # vmmd = VGAN(
        #     epochs=config.epochs, seed=config.seed, path_to_directory=config.path_to_directory,
        #     lr_G=config.lr_g, lr_D=config.lr_d, penalty=config.penalty, filename=config.filename,
        #     batch_size=config.batch_size, momentum=config.momentum, weight_decay=config.weight_decay,
        #     detector=config.detector, generator=config.generator, iternum_g=config.iternum_g, iternum_d=config.iternum_d,
        # )

        experiement = OutlierDetectionExperiment(
            vmmd=vmmd,
            od_model=CombinedOutlierDetector(
                base_estimators=[LOF()],
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
        #experiement.fit_pretrained_model("../experiments/remote/06-03/MVTEC_AD_normalize_imagesgen64_od256train256_lr=0.001_bs=1024_ep=2000_None/models/generator_2.pt")
        #experiement.fit_pretrained_model("../experiments/remote/07-03/OCCCIFAR10_normalize_images_train32_od32_lr=0.001_bs=1024_ep=2001_None/models/generator_9.pt")
        experiement.evaluate_interval(ensemble_weight_start=0, ensemble_weight_end=1, step= 1.0 / 10.0)
