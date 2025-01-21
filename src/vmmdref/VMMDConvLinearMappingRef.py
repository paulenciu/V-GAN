import os
import torch
from pathlib import Path


from src.vmmdref.VMMDRef import VMMDRef


class VMMDConvLinearMappingRef(VMMDRef):

    def __init__(self, filename="no_filename", batch_size=500, epochs=30, lr=0.007, momentum=0.99, seed=777, weight_decay=0.04,
             path_to_directory=Path(os.getcwd()).parent / "experiments" / "local", penalty_weight=0.0,):
        super().__init__(filename, batch_size, epochs, lr, momentum, seed, weight_decay, path_to_directory, False, penalty_weight)


    def sample_count_subspaces(self, count):
        return self._generate_subspaces(count, True)