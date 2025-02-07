import torch
import os

from pyod.models.lunar import LUNAR

from src.data.dataset_loader import load_data
from src.vmmd.penalty.MMDLossPenalty import MMDDiversityPenalty
from src.data.dataset_type import DatasetType
from src.models.encoder.IdentityEncoder import IdentityEncoder
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4 import GeneratorOneChannelV4
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV3 import GeneratorOneChannelV3

from src.outlier_detection import launch_outlier_detection_experiments
from src.vmmd.model.VMMDDiagonal1Channel import VMMDDiagonal1Channel
from src.vmmd.VMMDWrapper import VMMDWrapper

if __name__ == '__main__':
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()

    lr = 0.5
    epochs = 400
    batch_size = 512


    launch_outlier_detection_experiments(
        encoder=IdentityEncoder(),
        generator=GeneratorOneChannelV3(latent_size=100, image_shape=(1, 32, 32)),
        dataset_type=DatasetType.OCCFMNIST,
        normalize_data=True,
        image_size=(32, 32),
        category="Trouser",
        epochs=epochs,
        lr=lr,
        batch_size = batch_size,
        momentum = 0.8,
        weight_decay = 0.1,
        seed = 333,
        base_estimators=[LUNAR()],
        path_to_directory="../experiments/local",
        filename = f"v3_adad_benchmark_32_sm_wo_emb_occ_fmnist_1D_lr={lr}_ep={epochs}_bs={batch_size}",
        penalty=MMDDiversityPenalty(0)
    )

