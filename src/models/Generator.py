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