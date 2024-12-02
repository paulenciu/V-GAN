import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd

from src.models.Autoencoder import ResNet18AutoEncoder
from src.pipeline import Pipeline
from src.utils.L2DistanceUtility import plot_l2_distance
from src.utils.Plotter import plot_pvals, visualise_single_masking_of_vmmd, visualise_linear_mapping_of_vmmd, \
    visualise_rotations_of_vmmd
from src.vmmd.VMMDFlattened import VMMDFlattened
from src.vmmd.VMMDLinearMapping import VMMDLinearMapping
from src.vmmd.VMMDRotationMapping import VMMDRotationMapping
from src.vmmd.VMMDSingleMask import VMMDSingleMask

def load_vmmd(filepath, name="single"):
    if name == "single":
        vmmd = VMMDSingleMask()
        vmmd.load_models(path_to_generator=filepath, ndims=1024)
    elif name == "flatten":
        vmmd = VMMDFlattened()
        vmmd.load_models(path_to_generator=filepath, ndims=3072)
    elif name == "linear":
        vmmd = VMMDLinearMapping()
        vmmd.load_models(path_to_generator=filepath, ndims=32)
    else:
        raise NotImplementedError("Error, generator with label not found.")
    return vmmd

if __name__ == "__main__":
    lrs = [0.001, 0.0004]
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4955, 0.4564, 0.4155], std=[0.2568, 0.2523, 0.2580])
         ]
    )
    dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                           transform=transform)
    cats_dataset = [(img, label) for (img, label) in dataset if label == 3]

    #visualise_results("../experiments/single_masked_unscaled_resnet18_2024-11-25 09:32:54.536332_0.0001_2500/models/generator_0.pt")

    single_masked_vmmd = load_vmmd(
        "../experiments/single_masked_unscaled_resnet18_2024-11-25 09:32:54.536332_0.0001_2500/models/generator_0.pt",
                 name="single"
    )

    single_masked_vmmd2 = load_vmmd(
        "../experiments/single_masked_unscaled_resnet18_2024-11-24 09:59:39.127708_0.0001_3000/models/generator_0.pt",
        name="single"
    )

    vmmd1 = load_vmmd(
        "../experiments/unscaled_resnet18_2024-11-20 16:56:16.715808_0.0001_2000/models/generator_0.pt",
        name="flatten"
    )

    vmmd_linear = load_vmmd(
        "../experiments/linear_mapping_resnet18_2024-11-30 18:31:26.797733_0.001_200/models/generator_0.pt",
        name="linear"
    )

    vmmd_rot = VMMDRotationMapping()

    #print(vmmd_linear.generate_subspaces(1).to(torch.float32))

    #visualise_rotations_of_vmmd(vmmd_rot, 10)

    #plot_l2_distance(vmmd, cats_dataset, 3, single_mask=False, model_name="unscaled_resnet18")
    #plot_l2_distance(single_masked_vmmd, cats_dataset, upper_bound=300, single_mask=True, model_name="single_masked_unscaled_resnet18")
    #emb_func = ResNet18AutoEncoder().get_encoder().to(device)
    #plot_pvals(single_masked_vmmd2 ,x=cats_dataset, min=100, max=1000, step=1)
    #
    pl = Pipeline(
        lrs=lrs,
        n_epochs=200)
    pl.run(X=cats_dataset)







