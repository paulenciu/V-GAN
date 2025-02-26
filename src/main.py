import torch

from pyod.models.lunar import LUNAR
from src.run.OutlierDetectionExperiment import OutlierDetectionExperiment

from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4Softmax import GeneratorOneChannelV4Softmax

from src.od.CombinedOutlierDetector import CombinedOutlierDetector
from src.utils.preprocessing import normalize_images_col
from src.vmmd.model.VMMDDiagonal1Channel import VMMDDiagonal1Channel

from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty
from src.data.dataset_type import DatasetType
from src.models.encoder.IdentityEncoder import IdentityEncoder

if __name__ == '__main__':
    import os

    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()

    lr = 0.001
    latent_size= 128
    epochs = 1000
    batch_size = 1024
    store_stats = True
    penalty = MMDLossNoPenalty()
    momentum = 0.8
    weight_decay = 0.1
    standardize_data = False
    seed = 777
    n_channels = 3
    image_size = 64
    image_size_generator = (image_size, image_size)

    path_to_directory = "../experiments/remote"


    generator = GeneratorOneChannelV4Softmax(latent_size=latent_size, image_shape=(n_channels, image_size, image_size))
    encoder = IdentityEncoder()

    vmmd = VMMDDiagonal1Channel(
        epochs=epochs, seed=seed, path_to_directory=path_to_directory,
        lr=lr, penalty=penalty, filename=f"mvtec_train64_od128_1D_lr={lr}_ep={epochs}_bs={batch_size}",
        batch_size=batch_size, momentum=momentum, weight_decay=weight_decay,
        encoder=encoder, generator=generator
    )

    experiement = OutlierDetectionExperiment(
        vmmd=vmmd,
        od_model=CombinedOutlierDetector(base_estimators=[LUNAR()], vmmd=vmmd),
        dataset_type=DatasetType.MVTEC_AD,
        category=["bottle"],
        image_size_generator=image_size_generator,
        image_size_od=(128, 128),
        standardize_data=standardize_data,
        preprocessing_fn=normalize_images_col,
    )

    experiement.fit()
    for i in range(0, 10):
        experiement.evaluate(store_stats=True, weight_ensemble= i / 10.0)

    # epochs = 200
    # batch_size = 1500
    #
    # launch_outlier_detection_experiments(
    #     encoder=IdentityEncoder(),
    #     generator=GeneratorOneChannelV4Softmax(latent_size=latent_size, image_shape=(3, 32, 32)),
    #     dataset_type=DatasetType.OCCFMNIST,
    #     standardize_data=False,
    #     image_size=(32, 32),
    #     category="Trouser",
    #     epochs=epochs,
    #     lr=lr,
    #     skip_od=False,
    #     batch_size=batch_size,
    #     momentum=0.8,
    #     weight_decay=0.1,
    #     seed=333,
    #     base_estimators=[LUNAR()],
    #     path_to_directory="../experiments/remote",
    #     filename=f"fit_bw_adj_exp_scheduler_epochupdate_resized_normalized_unstandardized_mmd_v4_softmax_adam_32_occ_fmnist_1D_lr={lr}_ep={epochs}_bs={batch_size}",
    #     penalty=MMDLossNoPenalty(),
    # )
    #
    # launch_outlier_detection_experiments(
    #     encoder=IdentityEncoder(),
    #     generator=GeneratorOneChannelV4Softmax(latent_size=latent_size, image_shape=(3, 32, 32)),
    #     dataset_type=DatasetType.SYNTHETIC,
    #     standardize_data=False,
    #     image_size=(32, 32),
    #     category=["1", "2"],
    #     epochs=epochs,
    #     lr=lr,
    #     skip_od=False,
    #     batch_size=batch_size,
    #     momentum=0.8,
    #     weight_decay=0.1,
    #     seed=333,
    #     base_estimators=[LUNAR()],
    #     path_to_directory="../experiments/remote",
    #     filename=f"fit_bw_adj_exp_scheduler_epochupdate_resized_normalized_unstandardized_mmd_v4_softmax_adam_32_occ_syn_1D_lr={lr}_ep={epochs}_bs={batch_size}",
    #     penalty=MMDLossNoPenalty(),
    # )
    #
    #
    #
    #
    #
