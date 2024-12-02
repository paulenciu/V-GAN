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

def plot_l2_distance(vmmd, dataset, upper_bound, single_mask=False, model_name=None):
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    norms = {}
    for i in range(2, upper_bound):
        if single_mask:
            u = vmmd.generate_subspaces(i).repeat(1, 3)
        else:
            u = vmmd.generate_subspaces(i)
        big_u, _, _ = create_big_u(u)
        big_u = big_u.to(torch.float32).to(device)

        original = flatten_images_3d(dataset, device)
        masked = big_u * original

        distance: torch.Tensor = torch.sqrt(torch.sum(torch.abs(original - masked) ** 2))
        norms[i] = distance.to("cpu")
        print(i, distance)

    plot(x = list(norms.keys()),
         y = list(norms.values()),
         xlabel = "Number of sampled subspaces",
         ylabel = "Noise (L2 distance)",
         model_name = model_name
    )



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
