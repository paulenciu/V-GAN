import numpy as np
import torch


def create_big_u(u, n_masks):
    unique_subspaces, proba = np.unique(
        np.array(u.to('cpu')), axis=0, return_counts=True)
    proba = proba / np.array(u.to('cpu')).shape[0]

    unique_subspaces, proba = np.unique(
        np.array(u.to('cpu')), axis=0, return_counts=True)
    proba = proba / np.array(u.to('cpu')).shape[0]

    top_k_indices = np.argsort(proba)[-n_masks:][::-1]  # Sort in descending order

    # Extract the top k subspaces and their probabilities
    u = unique_subspaces[top_k_indices]
    top_k_probas = proba[top_k_indices]

    big_u = torch.tensor(np.zeros_like(unique_subspaces[0])).to(dtype=torch.float64)
    for i in range(len(unique_subspaces)):
        p = proba[i]
        s = torch.tensor(unique_subspaces[i]).to(dtype=torch.float64)
        big_u += p * s

    return big_u, u, proba