import numpy as np
import torch


def calculate_average_u(u):
    unique_subspaces, proba = np.unique(
        np.array(u.to('cpu')), axis=0, return_counts=True)
    proba = proba / np.array(u.to('cpu')).shape[0]

    big_u = torch.tensor(np.zeros_like(unique_subspaces[0])).to(dtype=torch.float64)
    for i in range(len(unique_subspaces)):
        p = proba[i]
        s = torch.tensor(unique_subspaces[i]).to(dtype=torch.float64)
        big_u += p * s

    return big_u