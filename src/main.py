import torch
import os

from pyod.models.lunar import LUNAR
from src.run.OutlierDetectionExperiment import OutlierDetectionExperiment

from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4Softmax import GeneratorOneChannelV4Softmax

from src.od.CombinedOutlierDetector import CombinedOutlierDetector
from src.utils.preprocessing import normalize_images_col, normalize_images_row, normalize_images_col_softmax
from src.vmmd.model.VMMDDiagonal1Channel import VMMDDiagonal1Channel

from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty
from src.data.dataset_type import DatasetType
from src.models.encoder.IdentityEncoder import IdentityEncoder

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


    lr = 0.001
    latent_size= 128
    epochs = 2000
    batch_size = 1024
    store_stats = True
    penalty = MMDLossNoPenalty()
    momentum = 0.8
    weight_decay = 0.1
    standardize_data = True
    seed = 333
    n_channels = 3

    image_size_generator = (32, 32)
    image_size_od = (32, 32)

    path_to_directory = "../experiments/remote"

    generator = GeneratorOneChannelV4Softmax(latent_size=latent_size, image_shape=(n_channels, *image_size_generator))
    encoder = IdentityEncoder()


    vmmd = VMMDDiagonal1Channel(
        epochs=epochs, seed=seed, path_to_directory=path_to_directory,
        lr=lr, penalty=penalty, filename=f"cifar_standardized_smcolnorm_train32_od32_1D_lr={lr}_ep={epochs}_bs={batch_size}",
        batch_size=batch_size, momentum=momentum, weight_decay=weight_decay,
        encoder=encoder, generator=generator
    )

    experiement = OutlierDetectionExperiment(
        vmmd=vmmd,
        od_model=CombinedOutlierDetector(base_estimators=[LUNAR()], vmmd=vmmd, max_n_jobs=1),
        dataset_type=DatasetType.OCCCIFAR10,
        category="cat",
        image_size_generator=image_size_generator,
        image_size_od=image_size_od,
        standardize_data=standardize_data,
        preprocessing_fn=normalize_images_col_softmax,
    )

    experiement.fit()
    experiement.evaluate_interval(ensemble_weight_start=0, ensemble_weight_end=1, step= 1.0 / 10.0)
