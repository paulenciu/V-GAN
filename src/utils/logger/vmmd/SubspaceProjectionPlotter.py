import os
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.utils.BigUBuilder import calculate_average_u
from src.utils.TensorConverter import tensor_to_image
from src.vmmd import VMMD
from src.utils.logger.vmmd.IVMMDLogger import IVMMDLogger
from src.vmmd.VMMDEmbedding import VMMDEmbedding


class SubspaceProjectionPlotter(IVMMDLogger):

    def __init__(self, vmmd: VMMD, base_dir: Path, n_samples=5, n_masks=5, sample_count=500):
        self.vmmd = vmmd
        self.n_samples = n_samples
        self.n_masks = n_masks
        self.base_dir = base_dir
        self.sample_count = sample_count

    def log(self, data, epoch=0):
        n_samples = self.n_samples
        n_masks = self.n_masks
        device = self.vmmd.device

        # Use LaTeX in the titles
        plt.rcParams['text.usetex'] = False
        fontsize = 80

        sample_indices = np.arange(n_samples)
        x_sample = torch.utils.data.Subset(data, sample_indices)

        n_channels, width, height = data.shape[1:]

        fig, axis = plt.subplots(n_samples + 1, 2 + n_masks, figsize=(5 * (2 + n_masks), 5 * (n_samples + 1)))
        u = self.vmmd.sample_count_subspaces(self.sample_count).to(device).detach()

        average_u, _, _ = calculate_average_u(u, n_masks)
        average_u = average_u.to(torch.float32).to(device)
        u = self.vmmd.sample_count_subspaces(n_masks).to(device)

        axis[0, 0].imshow(tensor_to_image(torch.ones(n_channels, height, width)))
        axis[0, 0].axis("off")

        if isinstance(self.vmmd, VMMDEmbedding):
            u_height = int(u.shape[1] / 224)
            u_width = int(u.shape[1] / u_height)


            u = u.view(-1, 1, u_width, u_height)
            u = u.repeat(1, 3, 1, 1) #makes image black / white

        for i in range(n_masks):
            axis[0, i + 1].imshow(tensor_to_image(u[i].detach()))
            axis[0, i + 1].axis("off")
            axis[0, i + 1].set_title(f"$U_{i + 1}$", fontsize=fontsize)

        if isinstance(self.vmmd, VMMDEmbedding):
            average_u = average_u.view(1, u_width, u_height)

        for i in range(1, n_samples + 1):

            image = x_sample[i - 1]
            image = image.to(torch.float32).to(device)

            axis[i, 0].imshow(tensor_to_image(image))

            if i == 1:
                axis[i, 0].set_title(f"Original", fontsize=fontsize)

            axis[i, 0].axis("off")

            u = self.vmmd.sample_count_subspaces(n_masks).to(device)
            image = image.unsqueeze(0).repeat(n_masks, 1, 1, 1).to(device)
            ux_data = self.vmmd.apply_subspaces_operator(image, u).to(device).detach()

            for j in range(n_masks):
                axis[i, j + 1].imshow(tensor_to_image(ux_data[j]))
                axis[i, j + 1].axis("off")

            big_u_image = self.vmmd.apply_subspaces_operator(u_subspaces=average_u.unsqueeze(0), x_sample=image[0]).squeeze(0)
            big_u_image = big_u_image.to(torch.float32).to(device)

            axis[i, n_masks + 1].imshow(tensor_to_image(big_u_image))
            axis[i, n_masks + 1].axis("off")

        if isinstance(self.vmmd, VMMDEmbedding):
            axis[0, n_masks + 1].imshow(tensor_to_image(average_u.repeat(3, 1, 1)))
        else:
            axis[0, n_masks + 1].imshow(tensor_to_image(average_u))

        axis[0, n_masks + 1].axis("off")
        axis[0, n_masks + 1].set_title(f"Average", fontsize=fontsize)

        plt.tight_layout()
        fig.subplots_adjust(wspace=0.05, hspace=0.05)

        plt.show()

        if self.base_dir is not None:

            path_to_plot_dir = self.base_dir / 'plots'
            if not path_to_plot_dir.exists():
                os.mkdir(path_to_plot_dir)
            run_number = int(len(os.listdir(path_to_plot_dir)))

            fig.savefig(path_to_plot_dir / f'plot_{run_number}.png')

