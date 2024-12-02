import torch


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import os
import operator
import torch.nn.functional as F


from src.models.Generator import GeneratorSingleMaskRes, GeneratorSingleMask, LinearMappingGenerator, \
    RotationalMatrixGenerator
from src.models.Mmd_loss_constrained import MMDLossConstrained, RBF
from src.utils.BigUBuilder import create_big_u
from src.utils.ImageFlattenerUtility import flatten_images_3d, unflatten_images_3d
from src.vmmd.vmmd import VMMD


class VMMDRotationMapping(VMMD):

    def __init__(self, batch_size=500, epochs=30, lr=0.007, momentum=0.99, seed=777, weight_decay=0.04,
                 path_to_directory=None):
        super().__init__(batch_size, epochs, lr, momentum, seed, weight_decay, path_to_directory)

    def rotate_images(self, rotation_matrices_2d, batch):

        rotation_matrices_2d = rotation_matrices_2d.to(self.device)
        batch = batch.to(self.device)

        n, c, h, w = batch.shape  # Example dimensions
        # Prepare affine matrices for grid sampling
        zeros = torch.zeros(n, 2, 1, device=self.device)
        affine_matrices = torch.cat([rotation_matrices_2d, zeros], dim=2)  # Shape: (n, 2, 3)

        # Generate grids
        grid = F.affine_grid(affine_matrices, batch.size(), align_corners=False)

        # Apply rotations to images
        rotated_images = F.grid_sample(batch, grid, align_corners=False, padding_mode='zeros')
        return rotated_images

    def load_models(self, path_to_generator, ndims, device: str = None):
        '''Loads models for prediction

        In case that the generator has already been trained, this method allows to load it (and optionally the discriminator) for generating subspaces
        Args:
            - path_to_generator: Path to the generator (has to be stored as a .keras model)
            - path_to_discriminator: Path to the discriminator (has to be stored as a .keras model) (Optional)
        '''
        if device == None:
            device = self.device
        self.__latent_size = max(int(3072 / 16), 1)
        self.generator = RotationalMatrixGenerator(
           latent_size=self.__latent_size).to(device)
        self.generator.load_state_dict(torch.load(path_to_generator))
        self.generator.eval()  # This only works for dropout layers
        self.generator_optimizer = f'Loaded Model from {path_to_generator} with {ndims} dimensions in the latent space'

    def fit(self, X, autoencoder):

        n_channels, width, height = X[0][0].shape[0], X[0][0].shape[1], X[0][0].shape[2]
        assert width == height, "Error, need square input images."

        encoder = autoencoder.get_encoder().to(self.device)

        X = flatten_images_3d(X, self.device)

        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()

        torch.manual_seed(self.seed)
        if cuda:
            torch.cuda.manual_seed(self.seed)
        elif mps:
            torch.mps.manual_seed(self.seed)

        # MODEL INTIALIZATION#
        epochs = self.epochs
        self.__latent_size = latent_size = max(int(X.shape[1] / 16), 1)
        ndims = X.shape[1]
        train_size = X.shape[0]
        self.batch_size = min(self.batch_size, train_size)

        X = unflatten_images_3d(X, channels=n_channels, width=width, height=height)

        generator = RotationalMatrixGenerator(latent_size).to(self.device)

        optimizer = torch.optim.Adadelta(
            generator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.generator_optimizer = optimizer.__class__.__name__

        loss_function = MMDLossConstrained(weight=10, kernel=RBF())

        for epoch in range(epochs):
            print(f'\rEpoch {epoch} of {epochs}')
            generator_loss = 0

            # DATA LOADER#
            if cuda:
                data_loader = DataLoader(
                    X, batch_size=self.batch_size, drop_last=True, pin_memory=cuda, shuffle=True)
            else:  # Uses CUDA if Available, other wise MPS or nothing
                data_loader = DataLoader(
                    X, batch_size=self.batch_size, drop_last=True, pin_memory=mps, shuffle=True)
            batch_number = data_loader.__len__()

            # GET NOISE TENSORS#
            if cuda:  # Need to open this if statement as the Tensor function has to be called from diferent modules depending of the device
                noise_tensor = torch.cuda.FloatTensor(
                    self.batch_size, latent_size).to(torch.device('cuda'))
            elif mps:
                noise_tensor = torch.mps.Tensor(
                    self.batch_size, latent_size).to(torch.device('mps'))
            else:
                noise_tensor = torch.Tensor(self.batch_size, latent_size)

            # BATCH LOOP#
            for batch in tqdm(data_loader, leave=False):
                # Make sure there is only 1 observation per row.
                # batch = batch.view(self.batch_size, -1)
                if cuda:
                    batch = batch.cuda()
                elif mps:
                    batch = batch.to(torch.float32).to(
                        torch.device('mps'))  # float64 not suported with mps

                # SAMPLE NOISE#
                noise_tensor.normal_()

                # OPTIMIZATION STEP#
                optimizer.zero_grad()
                rotation_matrices_2d = generator(noise_tensor).to(self.device)

                batch = batch.view(-1, n_channels, width, height)
                rotated_images = self.rotate_images(rotation_matrices_2d, batch)

                embedded_batch = encoder(batch)
                embedded_fake_subspaces = encoder(rotated_images)

                embedded_batch = embedded_batch.view(embedded_batch.size(0), -1)
                embedded_fake_subspaces = embedded_fake_subspaces.view(embedded_batch.size(0), -1)

                batch_loss = loss_function(embedded_batch, embedded_fake_subspaces, rotation_matrices_2d)
                self.bandwidth = loss_function.bandwidth
                batch_loss.backward()
                optimizer.step()
                generator_loss += float(batch_loss.to(
                    'cpu').detach().numpy()) / batch_number

            print(f"Average loss in the epoch: {generator_loss}")
            self.train_history["generator_loss"].append(generator_loss)

        if not self.path_to_directory == None:
            path_to_directory = Path(self.path_to_directory)
            if operator.not_(path_to_directory.exists()):
                os.mkdir(path_to_directory)
                if operator.not_(Path(path_to_directory / 'models').exists()):
                    os.mkdir(path_to_directory / 'models')
            run_number = int(len(os.listdir(path_to_directory / 'models')))
            torch.save(generator.state_dict(),
                       path_to_directory / 'models' / f'generator_{run_number}.pt')
            self.model_snapshot(path_to_directory, run_number, show=True)

        self.generator = generator

    def generate_subspaces(self, nsubs):
        # Need to load in cpu as mps Tensor module doesn't properly fix the seed
        #noise_tensor = torch.Tensor(nsubs, self.__latent_size).to('cpu') #FIXME commented out for testing purpose
        noise_tensor = torch.Tensor(nsubs, 192).to('cpu')
        self.generator = RotationalMatrixGenerator(192).to(self.device)
        if not self.seed == None:
            torch.manual_seed(self.seed)
        noise_tensor.normal_()
        u = self.generator(noise_tensor.to(self.device))
        return u
