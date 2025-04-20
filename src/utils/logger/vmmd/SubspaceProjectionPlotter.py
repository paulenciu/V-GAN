import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

import torchvision.transforms.functional
from src.utils.BigUBuilder import calculate_average_u
from src.utils.TensorConverter import tensor_to_image
from src.utils.utils import ycbcr_to_rgb, rgb_to_ycbcr, large_sobel, morphological_erosion
from src.vmmd import VMMD
from src.utils.logger.vmmd.IVMMDLogger import IVMMDLogger
from src.vmmd.VMMDEmbedding import VMMDEmbedding
from src.vmmd.VMMDEmbeddingSpace import VMMDEmbeddingSpace




def morphological_closing(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Performs morphological closing (dilation followed by erosion).
    mask: Tensor of shape (C,H,W) or (B,C,H,W), dtype bool or 0/1 float
    kernel_size: odd integer
    returns: closed mask, same shape and dtype
    """
    has_batch = mask.dim() == 4
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)  # (1,C,H,W)

    B, C, H, W = mask.shape
    device = mask.device
    pad = kernel_size // 2

    kernel = torch.ones((C, 1, kernel_size, kernel_size), device=device, dtype=torch.float32)

    m = mask.float()
    # Dilation
    dilated = F.conv2d(m, kernel, groups=C, padding=pad)
    dilated = (dilated > 0).float()
    # Erosion
    eroded = F.conv2d(dilated, kernel, groups=C, padding=pad)
    closed = (eroded == kernel_size * kernel_size)

    closed = closed.to(mask.dtype)
    if not has_batch:
        closed = closed.squeeze(0)

    return closed

def morphological_opening(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    mask: Bool or {0,1} Tensor of shape (C,H,W) or (B,C,H,W)
    kernel_size: odd integer
    returns: opened mask, same shape & dtype as input
    """
    # ensure batch dim
    has_batch = mask.dim() == 4
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)  # (1,C,H,W)

    B, C, H, W = mask.shape
    device = mask.device
    pad = kernel_size // 2

    # make per‑channel kernel
    kernel = torch.ones((C, 1, kernel_size, kernel_size), device=device, dtype=torch.float32)

    # erosion: all‑ones neighborhood → sum == kernel_size**2
    m = mask.float()
    eroded = F.conv2d(m, kernel, groups=C, padding=pad)
    eroded = (eroded == kernel_size * kernel_size).float()

    # dilation: any‑ones neighborhood → sum > 0
    dilated = F.conv2d(eroded, kernel, groups=C, padding=pad)
    opened = (dilated > 0)

    # restore original dtype & shape
    opened = opened.to(mask.dtype)
    if not has_batch:
        opened = opened.squeeze(0)  # (C,H,W)
    return opened

class SubspaceProjectionPlotter(IVMMDLogger):

    def __init__(self, vmmd: VMMD, base_dir: Path, n_samples=5, n_masks=5, sample_count=500):
        self.vmmd = vmmd
        self.n_samples = n_samples
        self.n_masks = n_masks
        self.base_dir = base_dir
        self.sample_count = sample_count

    def set_base_dir(self, base_dir: Path):
        self.base_dir = base_dir

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

        if isinstance(self.vmmd, VMMDEmbeddingSpace) and self.vmmd.decoder_available:
            fig, axis = plt.subplots(n_samples + 1, 3 + n_masks, figsize=(5 * (2 + n_masks), 5 * (n_samples + 1)))
        else:
            fig, axis = plt.subplots(n_samples + 1, 2 + n_masks, figsize=(5 * (2 + n_masks), 5 * (n_samples + 1)))

        u = self.vmmd.sample_count_subspaces(self.sample_count).to(device).detach()
        average_u = calculate_average_u(u)
        average_u = average_u.to(torch.float32).to(device)
        u = self.vmmd.sample_count_subspaces(n_masks).to(device)

        if isinstance(self.vmmd, VMMDEmbeddingSpace) and self.vmmd.decoder_available:
            u = torch.cat([torch.ones(1, u.shape[1]).to(u.device), u])
            n_masks += 1

        axis[0, 0].imshow(tensor_to_image(torch.ones(n_channels, height, width)))
        axis[0, 0].axis("off")

        if isinstance(self.vmmd, VMMDEmbedding) or isinstance(self.vmmd, VMMDEmbeddingSpace):
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

            image_clone = image.clone()
            #image_clone = torch.nn.functional.pad(image_clone, (2, 2, 2, 2), mode='replicate')  # for 5×5 kernel
            image_clone = torch.nn.functional.pad(image.clone(), (1, 1, 1, 1), mode='replicate')


            # Sobel X
            sobel_x = torch.tensor([
                                       [[[-1., 0., 1.],
                                         [-2., 0., 2.],
                                         [-1., 0., 1.]]]
                                   ] * 3).to(device)

            # Sobel Y
            sobel_y = sobel_x.transpose(-1, -2)

            # Sobel gradients
            grad_x = torch.nn.functional.conv2d(image_clone, sobel_x, padding=0, groups=3)
            grad_y = torch.nn.functional.conv2d(image_clone, sobel_y, padding=0, groups=3)
            sobel_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            sobel_mag = torch.abs(sobel_mag)
            sobel_mag = torch.mean(sobel_mag, dim=0).unsqueeze(0).repeat(3, 1, 1)
            mask = torch.greater(sobel_mag, 0.0)

            image = image * mask

            axis[i, 0].imshow(tensor_to_image(image))

            if i == 1:
                axis[i, 0].set_title(f"Original", fontsize=fontsize)

            axis[i, 0].axis("off")
            u = self.vmmd.sample_count_subspaces(n_masks).to(device)

            if isinstance(self.vmmd, VMMDEmbeddingSpace):
                if self.vmmd.decoder_available:
                    u = torch.cat([torch.ones(1, u.shape[1]).to(u.device), u])
                    image = image.unsqueeze(0).repeat(n_masks + 1, 1, 1, 1).to(device)
                    ux_data = self.vmmd.apply_subspaces_operator(image, u).to(device).detach()
                else:
                    break
            else:
                image = image.unsqueeze(0).repeat(n_masks, 1, 1, 1).to(device)
                ux_data = self.vmmd.apply_subspaces_operator(image, u).to(device).detach()

            for j in range(n_masks):
                ux_clone = ux_data[j].clone()
                image_clone = torch.nn.functional.pad(ux_clone, (1, 1, 1, 1), mode='replicate')

                # Sobel X
                sobel_x = torch.tensor([
                                           [[[-1., 0., 1.],
                                             [-2., 0., 2.],
                                             [-1., 0., 1.]]]
                                       ] * 3).to(device)

                # Sobel Y
                sobel_y = sobel_x.transpose(-1, -2)

                # Sobel gradients
                grad_x = torch.nn.functional.conv2d(image_clone, sobel_x, padding=0, groups=3)
                grad_y = torch.nn.functional.conv2d(image_clone, sobel_y, padding=0, groups=3)
                sobel_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
                sobel_mag = torch.abs(sobel_mag)
                sobel_mag = torch.mean(sobel_mag, dim=0).unsqueeze(0).repeat(3, 1, 1)
                mask = torch.greater(sobel_mag, 0.0)
                ux_clone = ux_data[j] * mask
                axis[i, j + 1].imshow(tensor_to_image(ux_clone))
                #axis[i, j + 1].imshow(tensor_to_image(ux_data[j]))
                axis[i, j + 1].axis("off")

            big_u_image = self.vmmd.apply_subspaces_operator(u_subspaces=average_u.unsqueeze(0), x_sample=image[0]).squeeze(0)
            big_u_image = big_u_image.to(torch.float32).to(device)

            axis[i, n_masks + 1].imshow(tensor_to_image(big_u_image))
            axis[i, n_masks + 1].axis("off")

        if isinstance(self.vmmd, VMMDEmbedding) or isinstance(self.vmmd, VMMDEmbeddingSpace):
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

