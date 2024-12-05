import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F



# Regular function definition does not appear to work properly within a Sequential definition of a network in Pytorchs
class upper_softmax(nn.Module):
    def __init__(self):
        super().__init__()  # Dummy intialization as there is no parameter to learn

    def forward(self, x):
        x = torch.nn.functional.softmax(x, 1)
        x = torch.less(x, 1/x.shape[1])*x + \
            torch.greater_equal(x, 1/x.shape[1])
        return x


class upper_lower_softmax(nn.Module):
    def __init__(self):
        super().__init__()  # Dummy intialization as there is no parameter to learn

    def forward(self, x):
        x = torch.nn.functional.softmax(x, 1)
        selected = torch.greater_equal(x, 1/x.shape[1])
        x = x*selected + (~selected)*1e-08
        return x

class Generator_copy(nn.Module):

    def __init__(self, original_generator):
        super(Generator_copy, self).__init__()
        self.model = original_generator
        self.softmax = upper_softmax()
        self.latent_size = original_generator.latent_size

        #resetting the parameters
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x

class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)

        )
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class GeneratorSingleMaskRes(nn.Module):
    def __init__(self, latent_size, img_size):
        super(GeneratorSingleMaskRes, self).__init__()

        # Initializing layers with residual blocks
        self.layers = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size),
            nn.BatchNorm1d(2 * latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * latent_size, 4* latent_size),
            nn.BatchNorm1d(4 * latent_size),
            nn.LeakyReLU(0.2),
            ResidualBlock2(4 * latent_size, 4 * latent_size),
            nn.Linear(4 * latent_size, 8 * latent_size),
            nn.BatchNorm1d(8 * latent_size),
            nn.LeakyReLU(0.2),
            ResidualBlock2(8 * latent_size, 8 * latent_size),
            nn.Linear(8 * latent_size, img_size),
            upper_softmax()
        )

    def forward(self, x):
        return self.layers(x)


class GeneratorSingleMask(nn.Module):
    def __init__(self, latent_size, img_size):
        super(GeneratorSingleMask, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size),
            nn.BatchNorm1d(2 * latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * latent_size, 4 * latent_size),
            nn.BatchNorm1d(4 * latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(4 * latent_size, 8 * latent_size),
            nn.BatchNorm1d(8 * latent_size),
            nn.Linear(8 * latent_size, 8 * latent_size),
            nn.BatchNorm1d(8 * latent_size),
            nn.Linear(8 * latent_size, img_size),
            upper_softmax()
        )

    def forward(self, input):
        x = self.layers(input)
        return x

class RotationalMatrixGenerator(nn.Module):
    def __init__(self, latent_size):
        super(RotationalMatrixGenerator, self).__init__()
        # Define the layers of the generator
        self.fc1 = nn.Linear(latent_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        # Pass the input through the layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # Normalize to ensure the vector lies on the unit circle
        rotation_matrix = self.f_function(x)
        return rotation_matrix

    def g_function(self, rotation_matrix):
        """
        Maps a 2D rotation matrix to the representation space.
        """
        return rotation_matrix[:, :, 0]  # Extract the first column (cos(theta), sin(theta))

    def f_function(self, vector):
        """
        Maps a 2D vector from the representation space back to a 2D rotation matrix.
        """
        vector = F.normalize(vector, dim=1)
        orthogonal_vector = torch.stack([-vector[:, 1], vector[:, 0]], dim=1)
        rotation_matrix = torch.stack([vector, orthogonal_vector], dim=2)
        return rotation_matrix

class RotationalMatrixGenerator2(nn.Module):
    def __init__(self, latent_size):
        super(RotationalMatrixGenerator2, self).__init__()
        # Define the layers of the generator
        self.fc1 = nn.Linear(latent_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 2)

    def forward(self, x):
        # Pass the input through the layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        # Normalize to ensure the vector lies on the unit circle
        rotation_matrix = self.f_function(x)
        return rotation_matrix

    def g_function(self, rotation_matrix):
        """
        Maps a 2D rotation matrix to the representation space.
        """
        return rotation_matrix[:, :, 0]  # Extract the first column (cos(theta), sin(theta))

    def f_function(self, vector):
        """
        Maps a 2D vector from the representation space back to a 2D rotation matrix.
        """
        norms = torch.norm(vector, dim=1, keepdim=True)

        # Normalize each vector
        normalized_vectors = vector / norms
        orthogonal_vector = torch.stack([-normalized_vectors[:, 1], normalized_vectors[:, 0]], dim=1)
        rotation_matrix = torch.stack([normalized_vectors, orthogonal_vector], dim=2)
        return rotation_matrix

class LinearMappingGenerator(nn.Module):
    def __init__(self, latent_size, h):
        super(LinearMappingGenerator, self).__init__()

        self.h = h

        self.network = nn.Sequential(
            nn.Linear(latent_size, 2*latent_size),
            nn.BatchNorm1d(2*latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(2*latent_size, 4*latent_size),
            nn.BatchNorm1d(4*latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(4*latent_size, 8*latent_size),
            nn.BatchNorm1d(8*latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(8*latent_size, 16*latent_size),
            nn.BatchNorm1d(16*latent_size),
            nn.Linear(16*latent_size, self.h * self.h),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.network(x)
        return x.view(x.size(0), self.h, self.h)


class LinearMappingGenerator2(nn.Module):
    def __init__(self, latent_size, h):
        super(LinearMappingGenerator2, self).__init__()

        self.h = h

        self.network = nn.Sequential(
            nn.Linear(latent_size, 2*latent_size),
            nn.BatchNorm1d(2*latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(2*latent_size, 4*latent_size),
            nn.BatchNorm1d(4*latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(4*latent_size, 8*latent_size),
            nn.BatchNorm1d(8*latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(8*latent_size, 16*latent_size),
            nn.BatchNorm1d(16*latent_size),
            nn.Linear(16*latent_size, self.h * self.h),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.network(x)
        return x.view(x.size(0), self.h, self.h)

class ConvLinearMappingGenerator(nn.Module):

    def __init__(self):
        super(ConvLinearMappingGenerator, self).__init__()

        self.noise_dim = torch.tensor([3, 1, 1])


        self.main = nn.Sequential(
            nn.ConvTranspose2d(3, 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()  #outputs to (-1, 1)
        )

    def forward(self, input):
        return self.main(input)


import torch
import torch.nn as nn


class ConvLinearMappingGeneratorSteroids(nn.Module):

    def __init__(self):
        super(ConvLinearMappingGeneratorSteroids, self).__init__()
        self.noise_dim = torch.tensor([1, 1, 1])

        self.main = nn.Sequential(
            # Input: (batch, 1, 1, 1)

            # First Transpose Convolution
            nn.ConvTranspose2d(1, 64, kernel_size=4, stride=1, padding=0, bias=False),  # Output: (batch, 64, 4, 4)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Second Transpose Convolution
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # Output: (batch, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Third Transpose Convolution
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # Output: (batch, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Fourth Transpose Convolution
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  # Output: (batch, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Output Layer
            nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False),  # Output: (batch, 1, 32, 32)
            nn.Tanh()  # Outputs to (-1, 1)
        )

        # Apply custom weight initialization
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, input):
        return self.main(input)

class ResidualBlock(nn.Module):
    """
    A residual block that adds the input to the output of a sequence of layers.
    This helps in training deeper networks by mitigating the vanishing gradient problem.
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.main(x))

class ConvLinearMappingGeneratorResnet(nn.Module):
    """
    The generator model that transforms a noise tensor into a mask tensor.
    """
    def __init__(self):
        super(ConvLinearMappingGeneratorResnet, self).__init__()

        self.noise_dim = torch.tensor([3, 3, 3])

        # Fully connected layer to expand the input noise
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 3, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        # Upsampling layers with residual blocks
        self.main = nn.Sequential(
            # Reshape to (batch_size, 512, 4, 4)
            ResidualBlock(512),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),  # (8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # (16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),   # (32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),              # Final output
            nn.Tanh()  # Outputs values between -1 and 1
        )

    def forward(self, input):
        # Flatten the input noise tensor
        x = input.view(input.size(0), -1)
        # Expand and reshape
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        # Generate the output mask
        x = self.main(x)
        return x
