import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd

from src.models.Autoencoder import ResNet18AutoEncoder
from src.pipeline import Pipeline
from src.utils.Plotter import visualise_conv_masking_of_vmmd, plot_pvals
from src.vmmd.VMMDConvLinearMapping import VMMDConvLinearMapping
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
    elif name == "rotation":
        vmmd = VMMDRotationMapping()
        vmmd.load_models(path_to_generator=filepath, ndims=2)
    elif name == "conv_linear":
        vmmd = VMMDConvLinearMapping()
        vmmd.load_models(path_to_generator=filepath, ndims=2)
    else:
        raise NotImplementedError("Error, generator with label not found.")
    return vmmd

if __name__ == "__main__":
    lrs = [0.4, 0.0004]
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

    vmmd_flatten = load_vmmd(
        "../experiments/unscaled_resnet18_2024-11-20 16:56:16.715808_0.0001_2000/models/generator_0.pt",
        name="flatten"
    )

    vmmd_linear = load_vmmd(
        "../experiments/linear_mapping_resnet18_2024-11-30 18:31:26.797733_0.001_200/models/generator_0.pt",
        name="linear"
    )

    vmmd_rot = load_vmmd(
    "../experiments/rotation_big_resnet18_2024-12-03 10:08:28.271269_0.001_200/models/generator_0.pt",
        name = "rotation"
    )

    # vmmd_conv_linear_new = load_vmmd(
    #     "../experiments/conv_linear_resnet18_2024-12-04 15:58:33.653160_0.01_200/models/generator_0.pt",
    #     name="conv_linear"
    # )
    # vmmd_conv_linear_old = load_vmmd(
    #     "../experiments/conv_linear_resnet18_2024-12-04 16:17:37.363859_0.1_1000/models/generator_0.pt",
    #     name="conv_linear"
    # )

    vmmd_conv_linear_1 = load_vmmd(
        "../experiments/conv_steroids_resnet18_2024-12-05 12:59:51.489610_0.1_200/models/generator_0.pt",
        name="conv_linear"
    )

    vmmd_conv_linear_2 = load_vmmd(
        "../experiments/conv_steroids_resnet18_2024-12-05 13:22:29.555671_0.0004_200/models/generator_0.pt",
        name="conv_linear"
    )

    # visualise_masking_of_vmmd(single_masked_vmmd, single=True)
    # visualise_masking_of_vmmd(single_masked_vmmd2, single=True)
    #
    # visualise_masking_of_vmmd(vmmd_flatten, single=False)
    #
   # visualise_rotations_of_vmmd(vmmd_rot)
    #
    # visualise_linear_mapping_of_vmmd(vmmd_linear, unsqueeze_fake_subspace=True)
    # visualise_conv_masking_of_vmmd(vmmd_conv_linear_steroids,
    #                                method="conv_linear",
    #                                n_masks=5,
    #                                path_to_experiment="../experiments/conv_steroids_resnet18_2024-12-04 19:13:34.901747_0.1_1000",
    #                                filename="big_conv_linear_image_0.1_1000")
    #
    # visualise_conv_masking_of_vmmd(vmmd_conv_linear_1,
    #                                method="conv_linear",
    #                                n_masks=5,
    #                                path_to_experiment="../experiments/conv_steroids_resnet18_2024-12-05 12:59:51.489610_0.1_200",
    #                                filename="conv_linear_image_0.1_200")
    #
    # visualise_conv_masking_of_vmmd(vmmd_conv_linear_2,
    #                                method="conv_linear",
    #                                n_masks=5,
    #                                path_to_experiment="../experiments/conv_steroids_resnet18_2024-12-05 13:22:29.555671_0.0004_200",
    #                                filename="conv_linear_image_0.0004_200")

    #upper_bound=300
    #plot_l2_masking_distance(single_masked_vmmd, cats_dataset, upper_bound, single_mask=True, model_name="single_masked1_unscaled_resnet18")
    #plot_l2_masking_distance(single_masked_vmmd2, cats_dataset, upper_bound, single_mask=True, model_name="single_masked2_unscaled_resnet18")
    #plot_l2_masking_distance(vmmd_flatten, cats_dataset, upper_bound, single_mask=False, model_name="flattend_unscaled_resnet18")


    min=100
    max=300
    step=1

    #plot_pvals(vmmd_rot ,x=cats_dataset, min=min, max=max, step=step, model="vmmd_rot_linear")

    #emb_func = ResNet18AutoEncoder().get_encoder().to(device)

    #plot_pvals(single_masked_vmmd ,x=cats_dataset, min=1000, max=1001, step=step, model="single_masked_vmmd")
    #plot_pvals(single_masked_vmmd2 ,x=cats_dataset, min=min, max=max, step=step, model="single_masked_vmmd2")


    # print(vmmd_rot.check_if_myopic(x_data=cats_dataset, emb_func=emb_func, count=1000))
    # print(vmmd_linear.check_if_myopic(x_data=cats_dataset, emb_func=emb_func, count=1000))
    # print(single_masked_vmmd.check_if_myopic(x_data=cats_dataset, emb_func=emb_func, count=1000))
    # print(single_masked_vmmd2.check_if_myopic(x_data=cats_dataset, emb_func=emb_func, count=1000))
    #print(vmmd_conv_linear_old.check_if_myopic(x_data=cats_dataset, emb_func=emb_func, count=1000))
    #print(vmmd_conv_linear_new.check_if_myopic(x_data=cats_dataset, emb_func=emb_func, count=1000))

    #plot_pvals(vmmd_conv_linear_steroids,x=cats_dataset, emb_func=emb_func, path_to_experiment="../experiments/conv_steroids_resnet18_2024-12-04 19:13:34.901747_0.1_1000")

    pl = Pipeline(
        lrs=lrs,
        n_epochs=100)
    pl.run(X=cats_dataset)







