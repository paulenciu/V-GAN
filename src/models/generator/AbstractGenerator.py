from abc import ABC, abstractmethod
import torch
from torch import nn

class AbstractGenerator(ABC, nn.Module):
    """
    Interface specifically for generator models.
    """

    def __init__(self):
        super().__init__()
        self._noise_dim = None
        self._img_shape = None

    @property
    def img_shape(self):
        """Abstract property for the noise dimension."""
        return self._img_shape

    @img_shape.setter
    def img_shape(self, value):
        """Abstract setter for the noise dimension."""
        self._img_shape = value

    @property
    def noise_dim(self):
        """Abstract property for the noise dimension."""
        return self._noise_dim

    @noise_dim.setter
    def noise_dim(self, value):
        """Abstract setter for the noise dimension."""
        self._noise_dim = value

    @abstractmethod
    def sample_subspace_masks(self, noise):
        """
        Passes the noise tensor to the subspace mask generator.

        Input: Noise tensor of get_noise_tensor_shape() shape
        Output: Subspace mask of dim (batch_num, 3, 32, 32).
        """
        pass
