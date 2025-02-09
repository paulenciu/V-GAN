import torch

from pyod.models.lunar import LUNAR
from src.models.generator.diagonal_matrix.three_channels.GeneratorThreeChannel import GeneratorThreeChannel

from src.models.generator.diagonal_matrix.three_channels.GeneratroOneChannelV10DBN import GeneratorThreeChannelV10DBN

from src.vmmd.penalty.MMDLossPenalty import MMDLossDiscretePenalty, MMDLossL2Penalty, MMDLossNoPenalty
from src.data.dataset_type import DatasetType
from src.models.encoder.IdentityEncoder import IdentityEncoder

from src.outlier_detection import launch_outlier_detection_experiments

if __name__ == '__main__':
    import os

    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()

    lr = 0.0001
    epochs = 100
    batch_size = 64

    launch_outlier_detection_experiments(
        encoder=IdentityEncoder(),
        generator=GeneratorThreeChannelV10DBN(latent_size=100, image_shape=(3, 32, 32), initial_temperature=0.1),
        dataset_type=DatasetType.SYNTHETIC,
        normalize_data=True,
        image_size=(32, 32),
        category=["1", "2"],
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        momentum=0.8,
        weight_decay=0.1,
        seed=333,
        base_estimators=[LUNAR()],
        path_to_directory="../experiments/remote",
        filename=f"od_v10_gn_dbn_adam_benchmark_32_sig_wo_emb_occ_syn_3D_lr={lr}_ep={epochs}_bs={batch_size}",
        penalty=MMDLossNoPenalty(),
    )

