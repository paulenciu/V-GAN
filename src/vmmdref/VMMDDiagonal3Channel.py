import os
from pathlib import Path

import torch

from src.utils.ImageFlattenerUtility import flatten_images_dataset_3d
from src.vmmdref.VMMDRef import VMMDRef
from src.vmmdref.penalty.MMDLossPenalty import MMDLossNoPenalty


class VMMDDiagonal3Channel(VMMDRef):

    def __init__(self, filename="no_filename", batch_size=500, epochs=30, lr=0.007, momentum=0.99, seed=777, weight_decay=0.04,
             path_to_directory=Path(os.getcwd()).parent / "experiments" / "local", penalty=MMDLossNoPenalty()):
        super().__init__(filename, batch_size, epochs, lr, momentum, seed, weight_decay, path_to_directory, False, penalty)

    def apply_subspaces_operator(self, x_sample_unflattened: torch.Tensor, u_subspaces: torch.Tensor):

        if len(x_sample_unflattened.shape) == 2:
            x_sample_flattened =  x_sample_unflattened.view(-1).to(self.device)
            u_subspaces_flattened = u_subspaces.view(-1).to(self.device)
        else:
            x_sample_flattened = x_sample_unflattened.view(x_sample_unflattened.shape[0], -1).to(self.device)
            u_subspaces_flattened = u_subspaces.view(u_subspaces.shape[0], -1).to(self.device)

        flattened_projection = u_subspaces_flattened * x_sample_flattened

        if len(x_sample_unflattened.shape) == 4:
            return flattened_projection.view(x_sample_unflattened.shape[0], x_sample_unflattened.shape[1] , x_sample_unflattened.shape[2], x_sample_unflattened.shape[3])

        if len(x_sample_unflattened.shape) == 2:
            return flattened_projection.unsqueeze(dim=0).view(1, x_sample_unflattened.shape[0], x_sample_unflattened.shape[1]).to(self.device)

        return flattened_projection.view(x_sample_unflattened.shape[0], x_sample_unflattened.shape[1] , x_sample_unflattened.shape[2])

    def _calculate_d(self, u):
        u = u.view(u.shape[0], -1)
        return u.shape[1]


    def sample_count_subspaces(self, count):
        return self._generate_subspaces(count, threshold=lambda u : 1 / self._calculate_d(u))
