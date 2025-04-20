import numpy as np
import torch
import math
import torchvision.transforms.functional as TF

# Example: Convert RGB to YCbCr and use chrominance (Cb, Cr)
import cv2

import torch
import numpy as np
import scipy.ndimage

# Generate 9x9 Sobel-like filter using Gaussian derivative
def large_sobel(size=5, axis=0):
    assert axis in [0, 1]
    # Create coordinate grid
    x = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, x)
    if axis == 0:
        kernel = scipy.ndimage.gaussian_filter1d(xx, sigma=1.5, order=1, axis=axis)
    else:
        kernel = scipy.ndimage.gaussian_filter1d(yy, sigma=1.5, order=1, axis=axis)
    return torch.tensor(kernel, dtype=torch.float32)



def rgb_to_ycbcr(images, normalize_y=False):
    """
    images: (B, 3, H, W), values in [0,1]
    Returns: (B, 3, H, W) tensor with Y, Cr, Cb channels, optionally normalized Y
    """
    B, _, H, W = images.shape
    images_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)  # (B, H, W, 3)

    ycbcr_list = []
    for img in images_np:
        ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)  # Y, Cr, Cb
        ycbcr_list.append(ycbcr)

    ycbcr_np = np.stack(ycbcr_list)  # (B, H, W, 3)
    ycbcr_tensor = torch.tensor(ycbcr_np).permute(0, 3, 1, 2).float() / 255.0  # (B, 3, H, W)

    if normalize_y:
        y = ycbcr_tensor[:, 0:1]  # Y channel
        y = (y - 0.5) / 0.5       # Normalize Y to [-1, 1]
        ycbcr_tensor = torch.cat([y, ycbcr_tensor[:, 1:]], dim=1)

    return ycbcr_tensor.to(images.device)


import cv2
import numpy as np
import torch

def ycbcr_to_rgb(ycbcr_tensor):
    """
    ycbcr_tensor: (B, 3, H, W) with Y, Cr, Cb in [0, 1]
    Returns: (B, 3, H, W) RGB tensor in [0, 1]
    """
    B, _, H, W = ycbcr_tensor.shape
    ycbcr_np = (ycbcr_tensor * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    rgb_list = []
    for ycrcb in ycbcr_np:
        rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        rgb_list.append(rgb)

    rgb_np = np.stack(rgb_list)  # (B, H, W, 3)
    rgb_tensor = torch.tensor(rgb_np).permute(0, 3, 1, 2).float() / 255.0  # (B, 3, H, W)
    return rgb_tensor.to(ycbcr_tensor.device)


def batch_crop_and_resize_softmax(softmax_batch, threshold=0.05):
    """
    softmax_batch: (B, C, H, W) tensor, assumed softmax probabilities
    threshold: threshold to mask out low-activation areas

    Returns:
        Tensor of shape (B, C, H, W) where low-activation areas are removed,
        content is cropped and resized back to (H, W)
    """
    B, C, H, W = softmax_batch.shape
    output = []
    threshold = 1.0 / math.prod(softmax_batch.shape[2:])
    for i in range(B):
        # Get max activation across channels
        activation_map = softmax_batch[i].max(dim=0)[0]  # (H, W)

        # Create binary mask of high-activation regions
        mask = activation_map > threshold

        # Get bounding box
        coords = mask.nonzero()
        if coords.shape[0] == 0:
            # No high-activation area, keep original
            cropped = softmax_batch[i]
        else:
            top_left = coords.min(0)[0]
            bottom_right = coords.max(0)[0] + 1

            # Crop and resize
            cropped = softmax_batch[i][:, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
            cropped = TF.resize(cropped, [H, W])

        output.append(cropped)

    return torch.stack(output)  # (B, C, H, W)

def morphological_erosion(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    has_batch = mask.dim() == 4
    if not has_batch:
        mask = mask.unsqueeze(0)  # (1,C,H,W)

    B, C, H, W = mask.shape
    pad = kernel_size // 2
    device = mask.device

    kernel = torch.ones((C, 1, kernel_size, kernel_size), device=device)
    m = mask.float()
    out = torch.nn.functional.conv2d(m, kernel, groups=C, padding=pad)
    eroded = (out == kernel_size * kernel_size)

    eroded = eroded.to(mask.dtype)
    if not has_batch:
        eroded = eroded.squeeze(0)
    return eroded
