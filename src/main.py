import torch
import os

from pyod.models.lunar import LUNAR
from src.run.OutlierDetectionExperiment import OutlierDetectionExperiment

from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4Softmax import GeneratorOneChannelV4Softmax

from src.od.CombinedOutlierDetector import CombinedOutlierDetector
from src.run.config.BaseConfiguration import BaseConfiguration
from src.run.config.TestConfiguration import TestConfiguration
from src.utils.preprocessing import normalize_features, normalize_images, normalize_images_col_softmax
from src.vmmd.model.VMMDDiagonal1Channel import VMMDDiagonal1Channel

from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty
from src.data.dataset_type import DatasetType
from src.models.encoder.IdentityEncoder import IdentityEncoder
from tqdm import tqdm


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
        # TestConfiguration(
        #     dataset_type=DatasetType.OCCCIFAR10,
        #     dateset_category="cat",
        # ),
        BaseConfiguration(
            dataset_type=DatasetType.OCCCIFAR10,
            dateset_category="cat",
            preprocessing_fn=normalize_features,
        ),
        BaseConfiguration(
            dataset_type=DatasetType.OCCFMNIST,
            dateset_category="Trouser",
            preprocessing_fn=normalize_features,
            image_size_od=(28, 28),
            image_size_generator=(28, 28),
        ),
        BaseConfiguration(
            dataset_type=DatasetType.MVTEC_AD,
            dateset_category=["bottle"],
            preprocessing_fn=normalize_features,
            image_size_od=(64, 64),
            image_size_generator=(128, 128),
        ),
        # BaseConfiguration(
        #     dataset_type=DatasetType.OCCCIFAR10,
        #     dateset_category="cat",
        #     preprocessing_fn=normalize_images,
        # ),
        # BaseConfiguration(
        #     dataset_type=DatasetType.OCCFMNIST,
        #     dateset_category="Trouser",
        #     preprocessing_fn=normalize_images,
        #     image_size_od=(28, 28),
        #     image_size_generator=(28, 28),
        # ),
        # BaseConfiguration(
        #     dataset_type=DatasetType.MVTEC_AD,
        #     dateset_category=["bottle"],
        #     preprocessing_fn=normalize_images,
        #     image_size_od=(64, 64),
        #     image_size_generator=(128, 128),
        # ),
        # BaseConfiguration(
        #     dataset_type=DatasetType.MVTEC_AD,
        #     dateset_category=["bottle"],
        #     preprocessing_fn=normalize_features,
        #     image_size_od=(64, 64),
        #     image_size_generator=(256, 256),
        # ),
        # BaseConfiguration(
        #     dataset_type=DatasetType.OCCFMNIST,
        #     dateset_category="Trouser",
        #     preprocessing_fn=normalize_images_col_softmax,
        #     image_size_od=(28, 28),
        #     image_size_generator=(28, 28),
        # ),
    ]

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
                base_estimators=[LUNAR()],
                vmmd=vmmd, max_n_jobs=1,
                preprocessing_fn=config.preprocessing_fn
            ),
            dataset_type=config.dataset_type,
            category=config.dateset_category,
            image_size_generator=config.image_size_generator,
            image_size_od=config.image_size_od,
            standardize_data=config.standardize_data,
            preprocessing_fn=config.preprocessing_fn,
            n_subspaces_sample=config.n_subspace_sample
        )

        experiement.fit()
        experiement.evaluate_interval(ensemble_weight_start=0, ensemble_weight_end=1, step= 1.0 / 10.0)
