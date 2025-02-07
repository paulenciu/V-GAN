import os
from pathlib import Path

from matplotlib import pyplot as plt

from src.utils.TensorConverter import tensor_to_image
from src.vmmd import VMMD
from src.vmmd.logger.ILogger import ILogger


class SubspaceDistributionPlotter(ILogger):

    def __init__(self, vmmd: VMMD, base_dir: Path, sample_count=500):
        self.vmmd = vmmd
        self.base_dir = base_dir
        self.sample_count = sample_count
        self.path_to_distribution_plots = self.__init_base_dir()


    def log(self, data):
        run_number = int(len(os.listdir(self.path_to_distribution_plots)))

        plot = self._create_mask_frequency_plot().to("cpu")

        plt.imshow(tensor_to_image(plot.detach()), cmap='hot', interpolation='nearest')
        plt.colorbar(label='Frequency of Masking')
        plt.title('Pixel Masking Distribution')
        plt.savefig(self.path_to_distribution_plots / f"freq_mask_{run_number}.pdf",
                    format="pdf", dpi=1200)
        plt.show()

    def __init_base_dir(self):
        path_to_distribution_plots = self.base_dir / "distribution"
        path_to_distribution_plots.mkdir(parents=True, exist_ok=True)
        return path_to_distribution_plots

    def _create_mask_frequency_plot(self):
        u = self.vmmd.sample_count_subspaces(self.sample_count)
        u_agg = u.sum(dim=0)
        return u_agg / u.shape[0]