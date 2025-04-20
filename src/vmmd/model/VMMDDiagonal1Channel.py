import os
from pathlib import Path

from src.vmmd.MMDLossConstrained import RBF
from src.vmmd.VMMD import VMMD
from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty



class VMMDDiagonal1Channel(VMMD):

    def __init__(self, autoencoder, generator, filename="no_filename", batch_size=500, epochs=30, lr=0.1, momentum=0.99, seed=777, weight_decay=0.04,
             path_to_directory=Path(os.getcwd()).parent / "experiments" / "local", penalty=MMDLossNoPenalty(), kernel=RBF()):
        super().__init__(filename, autoencoder, generator, batch_size, epochs, lr, momentum, seed, weight_decay, path_to_directory, penalty, kernel)

    def sample_count_subspaces(self, count):
        return self._generate_subspaces(count)