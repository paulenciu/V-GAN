import os
from pathlib import Path

import torch
from torch.utils.hipify.hipify_python import mapping
from src.vmmdref.VMMDRef import VMMDRef


class VMMDLinearMappingRef(VMMDRef):

    def __init__(self, filename="no_filename", batch_size=500, epochs=30, lr=0.007, momentum=0.99, seed=777, weight_decay=0.04,
             path_to_directory=Path(os.getcwd()).parent / "experiments" / "local"):
        super().__init__(filename, batch_size, epochs, lr, momentum, seed, weight_decay, path_to_directory, False)

    def sample_count_subspaces(self, count):
        #return self._generate_subspaces(count=count, generate_subspace_adjust=False) #FIXME only for testing
        return self._generate_subspaces(count=count, discretize_condition=True)


    def apply_subspaces_operator(self, x_sample_unflattened: torch.Tensor, u_subspaces: torch.Tensor):

        x_sample_unflattened = x_sample_unflattened.to(torch.float32)
        u_subspaces = u_subspaces.to(torch.float32)

        matmul_result = torch.matmul(u_subspaces, x_sample_unflattened)
        matmul_result = matmul_result - matmul_result.min()  # Shift minimum to 0
        matmul_result = matmul_result / matmul_result.max()  # Scale to [0, 1]

        return matmul_result