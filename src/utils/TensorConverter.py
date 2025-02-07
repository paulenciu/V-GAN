import numpy as np
import torch


def tensor_to_image(tensor):
    """
    Converts a PyTorch tensor to a numpy image array.
    Ensures that the dtype is compatible with matplotlib.
    """
    tensor = tensor.to(torch.float32)  # Ensure float32 type
    array = tensor.cpu().numpy()  # Convert to numpy
    array = np.clip(array, 0, 1)  # Ensure values are in [0, 1] range
    return array.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
