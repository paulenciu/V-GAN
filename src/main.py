import torch
import os

from src.data.dataset_loader import load_data
from src.data.dataset_type import DatasetType
from src.models.encoder.IdentityEncoder import IdentityEncoder
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4Extended import GeneratorOneChannelV4Extended
from src.vmmd.model.VMMDDiagonal1Channel import VMMDDiagonal1Channel
from src.vmmd.VMMDWrapper import VMMDWrapper

if __name__ == '__main__':
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()

    x_train, x_test, y_test = load_data(
        dataset_type=DatasetType.MVTEC_AD,
        category=["bottle"],
        image_size=(32, 32),
    )

    lr = 0.5
    epochs = 400
    batch_size = 512

    vmmd = VMMDDiagonal1Channel()
    vmmd_wrapper = VMMDWrapper(vmmd)

    vmmd.fit(
        dataset=x_train,
        encoder=IdentityEncoder(),
        generator=GeneratorOneChannelV4Extended(latent_size=100, image_shape=(3, 32, 32), temperature=3)
    )

    # launch_outlier_detection_experiments(
    #     encoder=IdentityEncoder(),
    #     generator=GeneratorOneChannelV4Extended(latent_size=100, image_shape=(3, 32, 32), temperature=3),
    #     dataset_type=DatasetType.SYNTHETIC,
    #     image_size=(32, 32),
    #     category="1",
    #     epochs=epochs,
    #     lr=lr,
    #     batch_size = batch_size,
    #     momentum = 0.8,
    #     weight_decay = 0.1,
    #     seed = 333,
    #     base_estimators=[LUNAR()],
    #     path_to_directory="../experiments/local",
    #     filename = f"v4_x_adam_benchmark_32_sm_wo_emb_occ_cifar10_1D_lr={lr}_ep={epochs}_bs={batch_size}",
    #     penalty=MMDDiversityPenalty(0)
    # )

