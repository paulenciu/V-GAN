import torch
import os

from pyod.models.lunar import LUNAR
from src.run.OutlierDetectionExperiment import OutlierDetectionExperiment

from src.od.CombinedOutlierDetector import CombinedOutlierDetector
from src.run.config.vgan.VGANTestConfiguration import VGANTestConfiguration
from src.run.config.vmmd.VMMDBaseConfiguration import VMMDBaseConfiguration
from src.utils.preprocessing import normalize_features
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
        VGANTestConfiguration()
    ]

    for i, config in enumerate(configs):

        print("RUNNING EXPERIMENT", i, " FROM", len(configs))

        # vmmd = VMMDDiagonal1Channel(
        #     epochs=config.epochs, seed=config.seed, path_to_directory=config.path_to_directory,
        #     lr=config.lr, penalty=config.penalty, filename=config.filename,
        #     batch_size=config.batch_size, momentum=config.momentum, weight_decay=config.weight_decay,
        #     encoder=config.encoder, generator=config.generator
        # )

        vgan = VGAN(
            epochs=config.epochs, seed=config.seed, path_to_directory=config.path_to_directory,
            lr_G=config.lr_g, lr_D=config.lr_d, penalty=config.penalty, filename=config.filename,
            batch_size=config.batch_size, momentum=config.momentum, weight_decay=config.weight_decay,
            detector=config.detector, generator=config.generator
        )

        experiement = OutlierDetectionExperiment(
            vmmd=vgan,
            od_model=CombinedOutlierDetector(
                base_estimators=[LUNAR()],
                vmmd=vgan, max_n_jobs=1,
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
