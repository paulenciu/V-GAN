import torch
from pyod.models.feature_bagging import FeatureBagging

from pyod.models.lunar import LUNAR
from src.models.encoder.pretrained_autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
from src.models.encoder.pretrained_autoencoder.resnet.ResNet50AutoEncoder import ResNet50AutoEncoder

from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4DBN import GeneratorOneChannelV4DBN
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4Softmax import GeneratorOneChannelV4Softmax
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4SoftmaxImproved import \
    GeneratorOneChannelV4DBNSoftmaxImproved

from src.models.generator.diagonal_matrix.three_channels.GeneratorThreeChannel import GeneratorThreeChannel

from src.models.generator.diagonal_matrix.three_channels.GeneratorThreeChannelV10DBNSA import GeneratorThreeChannelV10DBNSA
from src.models.generator.diagonal_matrix.three_channels.GeneratroOneChannelV10DBN import GeneratorThreeChannelV10DBN

from src.vmmd.penalty.MMDLossPenalty import MMDLossDiscretePenalty, MMDLossL2Penalty, MMDLossNoPenalty, \
    MMDDiversityPenalty
from src.data.dataset_type import DatasetType
from src.models.encoder.IdentityEncoder import IdentityEncoder

from src.outlier_detection import launch_outlier_detection_experiments, pretrained_launch_outlier_detection_experiments, \
    launch_outlier_detection_baseline

if __name__ == '__main__':
    import os

    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()

    lr = 0.001
    latent_size= 128
    epochs = 100
    batch_size = 64

    launch_outlier_detection_experiments(
        encoder=ResNet18AutoEncoder(),
        generator=GeneratorOneChannelV4Softmax(latent_size=latent_size, image_shape=(3, 32, 32)),
        dataset_type=DatasetType.SYNTHETIC,
        standardize_data=False,
        image_size=(32, 32),
        category=["1", "2"],
        epochs=epochs,
        lr=lr,
        skip_od=False,
        batch_size=batch_size,
        momentum=0,
        weight_decay=0.1,
        seed=333,
        base_estimators=[LUNAR()],
        path_to_directory="../experiments/local",
        filename="test",
        penalty=MMDLossNoPenalty(),
    )

    a = f"fit_bw_adj_exp_scheduler_epochupdate_resized_normalized_unstandardized_mmd_v4_softmax_adam_32_occ_mvtec_1D_lr={lr}_ep={epochs}_bs={batch_size}"

    epochs = 200
    batch_size = 1500

    launch_outlier_detection_experiments(
        encoder=IdentityEncoder(),
        generator=GeneratorOneChannelV4Softmax(latent_size=latent_size, image_shape=(3, 32, 32)),
        dataset_type=DatasetType.OCCFMNIST,
        standardize_data=False,
        image_size=(32, 32),
        category="Trouser",
        epochs=epochs,
        lr=lr,
        skip_od=False,
        batch_size=batch_size,
        momentum=0.8,
        weight_decay=0.1,
        seed=333,
        base_estimators=[LUNAR()],
        path_to_directory="../experiments/remote",
        filename=f"fit_bw_adj_exp_scheduler_epochupdate_resized_normalized_unstandardized_mmd_v4_softmax_adam_32_occ_fmnist_1D_lr={lr}_ep={epochs}_bs={batch_size}",
        penalty=MMDLossNoPenalty(),
    )

    launch_outlier_detection_experiments(
        encoder=IdentityEncoder(),
        generator=GeneratorOneChannelV4Softmax(latent_size=latent_size, image_shape=(3, 32, 32)),
        dataset_type=DatasetType.SYNTHETIC,
        standardize_data=False,
        image_size=(32, 32),
        category=["1", "2"],
        epochs=epochs,
        lr=lr,
        skip_od=False,
        batch_size=batch_size,
        momentum=0.8,
        weight_decay=0.1,
        seed=333,
        base_estimators=[LUNAR()],
        path_to_directory="../experiments/remote",
        filename=f"fit_bw_adj_exp_scheduler_epochupdate_resized_normalized_unstandardized_mmd_v4_softmax_adam_32_occ_syn_1D_lr={lr}_ep={epochs}_bs={batch_size}",
        penalty=MMDLossNoPenalty(),
    )





