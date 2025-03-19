import os
from pathlib import Path

import torch
import torch.nn.functional as F

from src.vmmd.VMMD import VMMD


class VMMDRotationMapping(VMMD):

    def __init__(self, filename="no_filename", batch_size=500, epochs=30, lr=0.007, momentum=0.99, seed=777, weight_decay=0.04,
             path_to_directory=Path(os.getcwd()).parent / "experiments" / "local"):
        super().__init__(filename, batch_size, epochs, lr, momentum, seed, weight_decay, path_to_directory, False)

    def apply_subspaces_operator(self, x_sample: torch.Tensor, u_subspaces: torch.Tensor):
        n, c, h, w = x_sample.shape  # Example dimensions
        # Prepare affine matrices for grid sampling
        zeros = torch.zeros(n, 2, 1, device=self.device)
        affine_matrices = torch.cat([u_subspaces, zeros], dim=2)  # Shape: (n, 2, 3)
        # Generate grids
        grid = F.affine_grid(affine_matrices, x_sample.size(), align_corners=False)

        # Apply rotations to images
        rotated_images = F.grid_sample(x_sample, grid, align_corners=False, padding_mode='zeros')
        return rotated_images

    def sample_count_subspaces(self, count):
        return self._generate_subspaces(count=count, discretize_condition=False)