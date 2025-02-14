import torch
from pyod.models.feature_bagging import FeatureBagging

from pyod.models.lunar import LUNAR

from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4DBN import GeneratorOneChannelV4DBN
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4DBNSoftmax import GeneratorOneChannelV4DBNSoftmax

from src.models.generator.diagonal_matrix.three_channels.GeneratorThreeChannel import GeneratorThreeChannel

from src.models.generator.diagonal_matrix.three_channels.GeneratorThreeChannelV10DBNSA import GeneratorThreeChannelV10DBNSA
from src.models.generator.diagonal_matrix.three_channels.GeneratroOneChannelV10DBN import GeneratorThreeChannelV10DBN

from src.vmmd.penalty.MMDLossPenalty import MMDLossDiscretePenalty, MMDLossL2Penalty, MMDLossNoPenalty
from src.data.dataset_type import DatasetType
from src.models.encoder.IdentityEncoder import IdentityEncoder

from src.outlier_detection import launch_outlier_detection_experiments, pretrained_launch_outlier_detection_experiments, \
    launch_outlier_detection_baseline

if __name__ == '__main__':
    import os

    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()

    lr = 0.0001
    epochs = 2000
    batch_size = 64

    launch_outlier_detection_experiments(
        encoder=IdentityEncoder(),
        generator=GeneratorOneChannelV4DBNSoftmax(latent_size=100, image_shape=(3, 64, 64), initial_temperature=0.1),
        dataset_type=DatasetType.OCCCIFAR10,
        normalize_data=False,
        image_size=(64, 64),
        category="cat",
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        momentum=0.8,
        weight_decay=0.1,
        seed=333,
        base_estimators=[LUNAR()],
        path_to_directory="../experiments/remote",
        filename=f"un_v4_softmax_dbn_adam_benchmark_64_wo_emb_occ_cifar10_1D_lr={lr}_ep={epochs}_bs={batch_size}",
        penalty=MMDLossNoPenalty(),
        skip_od=True
    )

    launch_outlier_detection_experiments(
        encoder=IdentityEncoder(),
        generator=GeneratorOneChannelV4DBNSoftmax(latent_size=100, image_shape=(3, 64, 64), initial_temperature=0.1),
        dataset_type=DatasetType.OCCFMNIST,
        normalize_data=False,
        image_size=(64, 64),
        category="Trouser",
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        momentum=0.8,
        weight_decay=0.1,
        seed=333,
        base_estimators=[LUNAR()],
        path_to_directory="../experiments/remote",
        filename=f"un_v4_softmax_dbn_adam_benchmark_64_wo_emb_occ_fmnist_1D_lr={lr}_ep={epochs}_bs={batch_size}",
        penalty=MMDLossNoPenalty(),
        skip_od=True
    )

    launch_outlier_detection_experiments(
        encoder=IdentityEncoder(),
        generator=GeneratorOneChannelV4DBNSoftmax(latent_size=100, image_shape=(3, 64, 64), initial_temperature=0.1),
        dataset_type=DatasetType.MVTEC_AD,
        normalize_data=False,
        image_size=(64, 64),
        category=["bottle"],
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        momentum=0.8,
        weight_decay=0.1,
        seed=333,
        base_estimators=[LUNAR()],
        path_to_directory="../experiments/remote",
        filename=f"un_v4_softmax_dbn_adam_benchmark_64_wo_emb_occ_mvtec_1D_lr={lr}_ep={epochs}_bs={batch_size}",
        penalty=MMDLossNoPenalty(),
        skip_od=True
    )