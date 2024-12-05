import numpy as np
import pandas as pd
import torch
import torch_two_sample as tts
import matplotlib.pyplot as plt

from src.utils.BigUBuilder import create_big_u
from src.utils.ImageFlattenerUtility import flatten_images_3d

def plot(x, xlabel, y, ylabel, model_name):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Model: " + model_name)
    plt.tight_layout()

    plt.show()

def plot_l2_masking_distance(vmmd, dataset, upper_bound, single_mask=False, model_name=None):
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    fig, axis = plt.subplots(1, 2, figsize=(15, 5))

    fig.suptitle("Model: " + model_name)

    reconstruction_norms = {}
    matrix_norm = {}
    for i in range(2, upper_bound):

        if single_mask:
            u = vmmd.generate_subspaces(i).repeat(1, 3)
        else:
            u = vmmd.generate_subspaces(i)

        big_u, _, _ = create_big_u(u)
        big_u = big_u.to(torch.float32).to(device)

        l2_norm = torch.norm(big_u, p='fro')
        matrix_norm[i] = l2_norm.to("cpu")

        original = flatten_images_3d(dataset).to(device)
        masked = big_u * original

        distance: torch.Tensor = torch.sqrt(torch.sum(torch.abs(original - masked) ** 2))
        reconstruction_norms[i] = distance.to("cpu")

    axis[0].plot(list(matrix_norm.keys()),list(matrix_norm.values()))
    axis[0].set_title("Matrix norm")
    axis[0].set_xlabel("Number of sampled subspaces")
    axis[0].set_ylabel("Noise (L2 distance)")

    axis[1].plot(list(reconstruction_norms.keys()), list(reconstruction_norms.values()))
    axis[1].set_title("image distance")
    axis[1].set_xlabel("Number of sampled subspaces")
    axis[1].set_ylabel("Noise (L2 distance)")

    plt.show()




def plot_l2_distance_of_mask(vmmd, upper_bound, single_mask=False, model_name=None):
    norms = {}
    for i in range(2, upper_bound):

        if single_mask:
            u = vmmd.generate_subspaces(i).repeat(1, 3)
        else:
            u = vmmd.generate_subspaces(i)

        big_u, _, _ = create_big_u(u)
        l2_norm = torch.norm(big_u, p='fro')
        norms[i] = l2_norm

    print("Big U", big_u)
    plot(x=list(norms.keys()),
         y=list(norms.values()),
         xlabel="Number of sampled subspaces",
         ylabel="Noise (L2 distance)",
         model_name=model_name
    )
