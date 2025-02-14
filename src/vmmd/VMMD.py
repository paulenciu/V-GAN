import re
import time

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union

import torch
from collections import defaultdict
from src.data.IDataset import IDataset

from sklearn.preprocessing import normalize
from src.utils.EMA import EMA
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import torch_two_sample as tts

from src.models.encoder.pretrained_autoencoder.AutoEncoderManager import AutoEncoderManager
from src.vmmd.logger.ILogger import ILogger
from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty
from src.models.Mmd_loss_constrained import MMDLossConstrained, RBF
from src.models.generator.AbstractGenerator import AbstractGenerator
from src.utils.BigUBuilder import calculate_average_u
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d, unflatten_images_3d
from src.vmmd.MMDLossConstrained import MMDLossConstrained


class VMMD(ABC):
    """
       V-MMD, a Subspace-Generative Moment Matching Network.

       Class for the method VMMD, the application of a GMMN to the problem of Subspace Generation. As a GMMN, no
       kernel learning is performed. The default values for the kernel are
    """

    def __init__(self, filename, batch_size=500, epochs=30, lr=0.007, momentum=0.99, seed=None, weight_decay=0.04,
                 path_to_directory= Path(os.getcwd()).parent / "experiments" / "local", penalty=MMDLossNoPenalty()):
        self.encoder = None
        self.generator = None
        self.penalty = penalty
        self.storage = locals()
        self.train_history = defaultdict(list)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.seed = seed
        self.weight_decay = weight_decay
        self.path_to_directory = path_to_directory
        self.generator_optimizer = None
        self.filename = filename
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.logger_subscriber: list[ILogger] = []

    def add_logger_subscriber(self, subscriber):
        self.logger_subscriber.append(subscriber)

    def notify_logging_subscriber(self, data: IDataset):
        for logger in self.logger_subscriber:
            logger.log(data)

    @abstractmethod
    def sample_count_subspaces(self, count):
        """
        Samples count many subspaces of shape (batch_size, channels, height, width) from the corresponding dataset.
        """
        pass

    def approx_subspace_dist(self, subspace_count=500):
        u = self.sample_count_subspaces(subspace_count)
        unique_subspaces, proba = np.unique(np.array(u.detach().to('cpu')), axis=0, return_counts=True)
        self.subspaces = torch.tensor(unique_subspaces).reshape(unique_subspaces.shape[0], -1).cpu().numpy()
        self.proba = proba / proba.sum()

    def load_model(self, generator: AbstractGenerator, autoencoder):
        if autoencoder is not None:
            self.encoder = autoencoder.to(self.device)

        self.generator = generator.to(self.device)
        self.generator.eval()

    def check_if_myopic(self, x_data: IDataset, bandwidth: Union[float, np.array] = 0.01, count=500) -> pd.DataFrame:
        """_summary_

        Args:
            x_data (np.array): Data to check the myopicity of.
            bandwidth (float | np.array, optional): Bandwidth used in the GOF tests using the MMD. This method always runs
            the recommended bandwidth alongside this optional one. Defaults to 0.01.
            count (int, optional): Number of samples used to approximate the MMD. Defaults to 500.

        Returns:
            pd.DataFrame: DataFrame containing the p.value of the test with all the different bandwidths.
        """
        if count > len(x_data):
            count = len(x_data)
        results = []

        n_channels, height, width = x_data.image_shape

        x_data = extract_and_flatten_images_dataset_3d(x_data).to("cpu")
        x_flattened_normalized = torch.from_numpy(normalize(x_data, axis=0)).to(torch.float32)
        x_sample = torch.Tensor(pd.DataFrame(x_flattened_normalized).sample(count).to_numpy()).to(self.device)

        del x_flattened_normalized, x_data

        with torch.no_grad():
            u_subspaces = self.sample_count_subspaces(count)
            x_sample = x_sample.view(-1, n_channels, height, width)
            ux_sample = self.apply_subspaces_operator(x_sample, u_subspaces).view(x_sample.shape[0], -1)
            x_sample = x_sample.view(x_sample.shape[0], -1)

            x_sample_embedded = self.encode(x_sample)
            ux_sample_embedded = self.encode(ux_sample)

        if type(bandwidth) == float:
            bandwidth = [bandwidth]

        if not hasattr(self, 'bandwidth'):
            mmd_loss = MMDLossConstrained()
            mmd_loss.forward(x_sample_embedded, ux_sample_embedded, u_subspaces * 1)
            self.bandwidth = mmd_loss.bandwidth

        bw = self.bandwidth.item()
        mmd = tts.MMDStatistic(count, count)
        _, distances = mmd(x_sample_embedded, ux_sample_embedded, alphas=[bw], ret_matrix=True)
        pval = mmd.pval(distances)
        results.append(pval)
        print("Count: ", count, "PVal: ", pval)
        # TODO wtf?
        results.append(pval)

        bandwidth.append("recommended bandwidth")
        return pd.DataFrame([results], columns=bandwidth, index=["p-val"])

    def apply_subspaces_operator(self, x_sample_unflattened: torch.Tensor, u_subspaces: torch.Tensor):
        return u_subspaces * x_sample_unflattened

    def encode(self, x):
        return self.encoder(x).view(x.shape[0], -1)

    def fit(self, dataset: IDataset, encoder, generator: AbstractGenerator):

        n_channels, width, height = dataset.image_shape
        assert width == height, "Error, need square input images."

        flattened_images = extract_and_flatten_images_dataset_3d(dataset).to("cpu")
        x_flattened_normalized = torch.from_numpy(normalize(flattened_images, axis=0)).to(torch.float32)
        unflattened_images = unflatten_images_3d(x_flattened_normalized, n_channels, height, width).to(self.device)

        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()

        torch.manual_seed(self.seed)
        if cuda:
            torch.cuda.manual_seed(self.seed)
        elif mps:
            torch.mps.manual_seed(self.seed)

        # MODEL INTIALIZATION#
        epochs = self.epochs
        train_size = flattened_images.shape[0]
        self.batch_size = min(self.batch_size, train_size)

        # SETUP GENERATOR OPTIMIZATION
        self.generator = generator.to(self.device)

        optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas = (0.5, 0.9),
            weight_decay=self.weight_decay
        )

        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        self.generator_optimizer = optimizer.__class__.__name__

        loss_function = MMDLossConstrained(penalty=self.penalty, kernel=RBF())

        snapshot_intervals = [int(i * 0.10 * epochs) for i in range(1, 11)]

        self.encoder = encoder
        self.encoder.eval()
        self.encoder = self.encoder.to(self.device)

        #INITIAL SNAPSHOT
        self.notify_logging_subscriber(dataset)

        # DATA LOADER#
        data_loader = self.__setup_data_loader(unflattened_images, cuda, mps)
        batch_number = data_loader.__len__()

        print("Device used:", self.device)

        # GET NOISE TENSORS#
        noise = self.__setup_noise_tensor(generator_input_shape=generator.noise_dim)

        total_training_time = 0.0
        snapshot_duration = 0.0

        for epoch in range(epochs):
            print(f'\rEpoch {epoch} of {epochs}')
            generator_loss = 0
            mmd_loss_avg = 0
            epoch_start = time.time()
            # BATCH LOOP#
            for batch in tqdm(data_loader, leave=False):

                if cuda:
                    batch = batch.cuda()
                elif mps:
                    batch = batch.to(torch.float32).to(torch.device('mps'))  # float64 not suported with mps

                # SAMPLE NOISE#
                noise.normal_()

                # OPTIMIZATION STEP#
                optimizer.zero_grad()

                u_mappings = generator.sample_subspace_masks(noise).to(self.device)
                processed_batch = self.apply_subspaces_operator(batch, u_mappings)

                #Encoder assumes a 3 channeled input
                if batch.shape[1] == 1:
                    batch = batch.repeat(1, 3, 1, 1)
                    processed_batch = processed_batch.repeat(1, 3, 1, 1)

                embedded_batch = self.encode(batch).to(self.device)
                embedded_processed_batch = self.encode(processed_batch).to(self.device)

                batch_loss, mmd_loss = loss_function(embedded_batch, embedded_processed_batch, u_mappings)

                self.bandwidth = loss_function.bandwidth
                batch_loss.backward()
                optimizer.step()
                scheduler.step()

                generator_loss += batch_loss.item() / batch_number
                mmd_loss_avg += mmd_loss.item() / batch_number

                # Clean up intermediate variables
                del u_mappings, processed_batch, embedded_batch, embedded_processed_batch

            generator.anneal_temperature()

            epoch_duration = time.time() - epoch_start
            total_training_time += epoch_duration
            self.train_history["training_time"] = total_training_time

            #INBETWEEN SNAPSHOTS
            if epoch in snapshot_intervals:
                self.train_history["training_time"] -= snapshot_duration
                snapshot_start = time.time()
                self.notify_logging_subscriber(dataset)
                snapshot_duration = time.time() - snapshot_start

            print(f"Average loss in the epoch: {generator_loss}")
            self.train_history["generator_loss"].append(generator_loss)
            self.train_history["mmd_loss"].append(mmd_loss_avg)

        generator.anneal_temperature()

        #FINAL SNAPSHOT
        self.notify_logging_subscriber(dataset)

        self.train_history["training_time"] = total_training_time

        self.generator = generator

    def __setup_data_loader(self, x_unflattened, cuda, mps):
        if cuda:
            return DataLoader(x_unflattened, batch_size=self.batch_size, drop_last=True, pin_memory=False, shuffle=True)
        else:  # Uses CUDA if available, otherwise MPS or nothing
            return DataLoader(x_unflattened, batch_size=self.batch_size, drop_last=True, pin_memory=mps, shuffle=True)


    def _generate_subspaces(self, count, threshold = None):

        generator_input_shape = self.generator.noise_dim

        # Need to load in cpu as mps Tensor module doesn't properly fix the seed
        noise_tensor = self.__setup_noise_tensor(batch_size=count, generator_input_shape=generator_input_shape)

        if not self.seed is None:
            torch.manual_seed(self.seed)

        noise_tensor.normal_()
        u: torch.Tensor = self.generator.sample_subspace_masks(noise_tensor.to(self.device), mode="test")
        return u

    def __setup_noise_tensor(self, generator_input_shape: torch.Tensor, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        shape = (batch_size, *generator_input_shape)
        return torch.empty(shape, dtype=torch.float32, device=self.device)