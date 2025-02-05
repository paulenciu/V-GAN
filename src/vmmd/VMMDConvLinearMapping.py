import os
import torch
from pathlib import Path


from src.vmmd.VMMD import VMMD
from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty


class VMMDConvLinearMapping(VMMD):

    def __init__(self, filename="no_filename", batch_size=500, epochs=30, lr=0.007, momentum=0.99, seed=777, weight_decay=0.04,
             path_to_directory=Path(os.getcwd()).parent / "experiments" / "local", penalty=MMDLossNoPenalty()):
        super().__init__(filename, batch_size, epochs, lr, momentum, seed, weight_decay, path_to_directory, False, penalty)


    def sample_count_subspaces(self, count):
        return self._generate_subspaces(count, threshold=lambda u : 1 / 2)