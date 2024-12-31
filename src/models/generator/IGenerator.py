from abc import ABC, abstractmethod

from torch import nn



class IGenerator(ABC, nn.Module):
    """
    Interface specifically for generator models.
    """

    @abstractmethod
    def get_noise_tensor_shape(self):
        """
        Returns the shape of the noise tensor.
        """
        pass

    @abstractmethod
    def sample_subspace_masks(self, noise):
        """
        Passes the noise tensor to the subspace mask generator.

        Input: Noise tensor of get_noise_tensor_shape() shape
        Output: Subspace mask of dim (batch_num, 3, 32, 32).
        """
        pass