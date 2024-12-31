import torch

from src.models.autoencoder.resnet.RestNetAutoEncoder import ResNet18AutoEncoder
from src.utils.Plotter import visualise_linear_mapping_of_vmmd
from src.utils.VMMDLoader import load_vmmd

if __name__ == "__main__":
    lrs = [0.1, 0.0004]
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')


    emb_func = ResNet18AutoEncoder().get_encoder().to(device)

    vmmd_linear = load_vmmd(
        filepath="../experiments/remote/experiments/linear_mapping_resnet18_2024-12-06 09:57:00.613177_0.5_500/models/generator_0.pt",
        name="linear"
    )

    visualise_linear_mapping_of_vmmd(vmmd_linear)




