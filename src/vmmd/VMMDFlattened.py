from typing import Union

import torch
from collections import defaultdict


from models.Mmd_loss_constrained import MMDLossConstrained, RBF
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import operator
import datetime
import torch_two_sample as tts

from src.models.generator.Generator5_16 import Generator5_16
from src.utils.BigUBuilder import create_big_u
from src.utils.ImageFlattenerUtility import flatten_images_3d, unflatten_images_3d
from src.vmmd.vmmd import VMMD


class VMMDFlattened(VMMD):
    '''
    V-MMD, a Subspace-Generative Moment Matching Network.

    Class for the method VMMD, the application of a GMMN to the problem of Subspace Generation. As a GMMN, no
    kernel learning is performed. The default values for the kernel are
    '''

    def __init__(self, batch_size=500, epochs=30, lr=0.007, momentum=0.99, seed=777, weight_decay=0.04,
                 path_to_directory=None):
        super().__init__(batch_size, epochs, lr, momentum, seed, weight_decay, path_to_directory)

    def check_if_myopic(self, x_data, bandwidth: Union[float, np.array] = 0.01, count=500) -> pd.DataFrame:
        """_summary_

        Args:
            x_data (np.array): Data to check the myopicity of.
            bandwidth (float | np.array, optional): Bandwidth used in the GOF tests using the MMD. This method always runs
            the recommended bandwidth alongside this optional one. Defaults to 0.01.
            count (int, optional): Number of samples used to approximate the MMD. Defaults to 500.

        Returns:
            pd.DataFrame: DataFrame containing the p.value of the test with all the different bandwidths.
        """
        assert count <= len(x_data), "Selected 'count' is greater than the number of samples in the dataset"
        results = []

        x_data = flatten_images_3d(x_data, "cpu")

        x_data = normalize(x_data, axis=0)
        x_sample = torch.Tensor(pd.DataFrame(
            x_data).sample(count).to_numpy()).to(self.device)
        u_subspaces = self.generate_subspaces(count)
        big_u, _, _ = create_big_u(u_subspaces)
        big_u = big_u.to(torch.float32).to(self.device)

        ux_sample = big_u * x_sample

        if type(bandwidth) == float:
            bandwidth = [bandwidth]

        if not hasattr(self, 'bandwidth'):
            mmd_loss = MMDLossConstrained(0)
            mmd_loss.forward(
                x_sample, ux_sample, u_subspaces * 1)
            self.bandwidth = mmd_loss.bandwidth

        # bandwidth.sort()
        # for bw in bandwidth:
        #     mmd = tts.MMDStatistic(count, count)
        #     _, distances = mmd(x_sample, ux_sample, alphas=[
        #         bw], ret_matrix=True)
        #     pval = mmd.pval(distances)
        #     print("Count: ", count, "PVal: " ,pval)
        #     results.append(pval)

        bw = self.bandwidth.item()
        mmd = tts.MMDStatistic(count, count)
        _, distances = mmd(x_sample, ux_sample, alphas=[bw], ret_matrix=True)
        pval = mmd.pval(distances)
        results.append(pval)
        print("Count: ", count, "PVal: ", pval)
        results.append(pval)

        bandwidth.append("recommended bandwidth")
        return pd.DataFrame([results], columns=bandwidth, index=["p-val"])

    def load_models(self, path_to_generator, ndims, device: str = None):
        '''Loads models for prediction

        In case that the generator has already been trained, this method allows to load it (and optionally the discriminator) for generating subspaces
        Args:
            - path_to_generator: Path to the generator (has to be stored as a .keras model)
            - path_to_discriminator: Path to the discriminator (has to be stored as a .keras model) (Optional)
        '''
        if device == None:
            device = self.device
        self.generator = Generator5_16(
            img_size=ndims, latent_size=max(int(ndims/16), 1)).to(device)
        self.generator.load_state_dict(torch.load(path_to_generator))
        self.generator.eval()  # This only works for dropout layers
        self.generator_optimizer = f'Loaded Model from {path_to_generator} with {ndims} dimensions in the latent space'
        self.__latent_size = max(int(ndims/16), 1)

    def fit(self, X, autoencoder, generator="generator_s"):

        encoder = autoencoder.get_encoder().to(self.device)

        n_channels, width, height = X[0][0].shape[0], X[0][0].shape[1], X[0][0].shape[2]
        X = flatten_images_3d(X)

        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()

        torch.manual_seed(self.seed)
        if cuda:
            torch.cuda.manual_seed(self.seed)
        elif mps:
            torch.mps.manual_seed(self.seed)

        # MODEL INTIALIZATION#
        epochs = self.epochs
        self.__latent_size = latent_size = max(int(X.shape[1]/16), 1)
        ndims = X.shape[1]
        train_size = X.shape[0]
        self.batch_size = min(self.batch_size, train_size)

        device = self.device

        if generator == "generator_s":
            generator = Generator5_16(latent_size=latent_size, img_size=ndims).to(device)
        elif generator == "generator_h":
            generator = GeneratorHierarchical(latent_size=latent_size, img_size=ndims).to(device)
        else:
            raise NotImplementedError("Error, generator not implemented.")

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
                batch = batch.view(self.batch_size, -1)
                if cuda:
                    batch = batch.cuda()
                elif mps:
                    batch = batch.to(torch.float32).to(
                        torch.device('mps'))  # float64 not suported with mps

                # SAMPLE NOISE#
                noise_tensor.normal_()

                # OPTIMIZATION STEP#
                optimizer.zero_grad()
                fake_subspaces = generator(noise_tensor)
                processed_fake_subspaces = fake_subspaces * batch + torch.less(
                    batch, 1 / batch.shape[1]) * torch.mean(batch, dim=0)

                processed_fake_subspaces = unflatten_images_3d(processed_fake_subspaces, n_channels, width, height)
                batch = unflatten_images_3d(batch, n_channels, width, height)

                embedded_batch = encoder(batch)
                embedded_fake_subspaces = encoder(processed_fake_subspaces)

                embedded_batch = embedded_batch.view(embedded_batch.size(0), -1)
                embedded_fake_subspaces = embedded_fake_subspaces.view(embedded_batch.size(0), -1)

                batch_loss = loss_function(embedded_batch, embedded_fake_subspaces, fake_subspaces)
                self.bandwidth = loss_function.bandwidth
                batch_loss.backward()
                optimizer.step()
                generator_loss += float(batch_loss.to(
                    'cpu').detach().numpy())/batch_number

            print(f"Average loss in the epoch: {generator_loss}")
            self.train_history["generator_loss"].append(generator_loss)

        if not self.path_to_directory == None:
            path_to_directory = Path(self.path_to_directory)
            if operator.not_(path_to_directory.exists()):
                os.mkdir(path_to_directory)
                if operator.not_(Path(path_to_directory/'models').exists()):
                    os.mkdir(path_to_directory / 'models')
            run_number = int(len(os.listdir(path_to_directory/'models')))
            torch.save(generator.state_dict(),
                       path_to_directory/'models'/f'generator_{run_number}.pt')
            self.model_snapshot(path_to_directory, run_number, show=True)

        self.generator = generator

    def generate_subspaces(self, nsubs):
        # Need to load in cpu as mps Tensor module doesn't properly fix the seed
        noise_tensor = torch.Tensor(nsubs, self.__latent_size).to('cpu')
        if not self.seed == None:
            torch.manual_seed(self.seed)
        noise_tensor.normal_()
        u = self.generator(noise_tensor.to(self.device))
        u = torch.greater_equal(u, 1/u.shape[1])
        return u
