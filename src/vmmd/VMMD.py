import time

from abc import ABC, abstractmethod
from typing import Union

import torch
from collections import defaultdict
from src.data.IDataset import IDataset

from sklearn.preprocessing import normalize
from src.data.dataset.PreEmbeddedDataset import PreEmbeddedDataset
from src.utils.preprocessing import normalize_features
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import os
import torch_two_sample as tts

from src.utils.logger.vmmd.IVMMDLogger import IVMMDLogger
from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty
from src.models.generator.AbstractGenerator import AbstractGenerator
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d, unflatten_images_3d
from src.vmmd.MMDLossConstrained import MMDLossConstrained, MixtureRQLinear, RBF


class VMMD(ABC):
    """
       V-MMD, a Subspace-Generative Moment Matching Network.

       Class for the method VMMD, the application of a GMMN to the problem of Subspace Generation. As a GMMN, no
       kernel learning is performed. The default values for the kernel are
    """

    def __init__(self, filename, autoencoder, generator: AbstractGenerator, batch_size=500, epochs=30, lr=0.007, momentum=0.99, seed=None, weight_decay=0.04,
                 path_to_directory= Path(os.getcwd()).parent / "experiments" / "local", penalty=MMDLossNoPenalty()):
        self.autoencoder = autoencoder
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
        self.encoder = autoencoder.get_encoder_and_freeze()
        self.generator = generator
        self.filename = filename
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.logger_subscriber: list[IVMMDLogger] = []

    @abstractmethod
    def sample_count_subspaces(self, count):
        """
        Samples count many subspaces of shape (batch_size, channels, height, width) from the corresponding dataset.
        """
        pass

    def approx_subspace_dist(self, subspace_count=500):
        u = self.sample_count_subspaces(subspace_count)
        unique_subspaces, proba = np.unique(np.array(u.detach().to('cpu')), axis=0, return_counts=True)
        self.subspaces = torch.Tensor(unique_subspaces)
        self.proba = torch.Tensor(proba / proba.sum())

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
        count = min(count, len(x_data))
        results = []

        n_channels, height, width = x_data.image_shape

        with torch.no_grad():
            x_sample = next(iter(DataLoader(x_data, batch_size=count)))[0].to(self.device)
            noise = torch.randn(count, *self.generator.noise_dim, device=self.device)
            u_mappings = self.generator.sample_subspace_masks(noise)

            ux_sample = self.apply_subspaces_operator(x_sample, u_mappings)

            x_embeddings = self.encode(x_sample)
            ux_embeddings = self.encode(ux_sample)




        # x_data = extract_and_flatten_images_dataset_3d(x_data).to("cpu")
        # x_sample = torch.Tensor(pd.DataFrame(x_data).sample(count).to_numpy()).to(self.device)
        # x_sample = x_sample.view(-1, n_channels, height, width)
        #
        # u_mappings = self.sample_count_subspaces(count).to(torch.float32)
        # ux_sample = self.apply_subspaces_operator(x_sample, u_mappings)
        #
        # x_sample_embedded = self.encode(x_sample)
        # ux_sample_embedded = self.encode(ux_sample)

        if type(bandwidth) == float:
            bandwidth = [bandwidth]

        mmd_loss = MMDLossConstrained()
        mmd_loss.forward(x_embeddings, ux_embeddings, u_mappings * 1)
        self.bandwidth = mmd_loss.bandwidth

        bw = self.bandwidth.item()
        print("Bw: ", bw)
        mmd = tts.MMDStatistic(count, count)
        _, distances = mmd(x_embeddings, ux_embeddings, alphas=[bw], ret_matrix=True)
        pval = mmd.pval(distances)
        results.append(pval)
        print("Count: ", count, "PVal: ", pval)
        # TODO wtf?
        results.append(pval)

        bandwidth.append("recommended bandwidth")
        return pd.DataFrame([results], columns=bandwidth, index=["p-val"])

    def apply_subspaces_operator(self, x_sample: torch.Tensor, u_subspaces: torch.Tensor):
        u_subspaces = u_subspaces.to(torch.float32)

        if x_sample.shape[2] != u_subspaces.shape[2]:
            u_subspaces = interpolate(u_subspaces, size=x_sample.shape[2], mode="nearest")

        return u_subspaces * x_sample

    def encode(self, x):
        return self.encoder(x).view(x.shape[0], -1)

    def setup_device_and_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        elif torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)

    def setup_optimizer_and_scheduler(self):
        optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(0.5, 0.9),
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return optimizer, scheduler

    def setup_data_loader(self, dataset, preprocess_fn=normalize_features, **preprocess_kwargs):
        n_channels, height, width = dataset.image_shape
        flattened_images = extract_and_flatten_images_dataset_3d(dataset).cpu()
        x_flattened_preprocessed = torch.from_numpy(
            preprocess_fn(flattened_images.numpy(), **preprocess_kwargs)).float()
        unflattened_images = unflatten_images_3d(x_flattened_preprocessed, n_channels, height, width)
        pre_embedded_dataset = PreEmbeddedDataset(unflattened_images, self.encoder, self.device)
        return DataLoader(
            pre_embedded_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

    # def setup_data_loader(self, dataset, n_channels, height, width, preprocess_fn=normalize_features,
    #                       **preprocess_kwargs):
    #     flattened_images = extract_and_flatten_images_dataset_3d(dataset).to("cpu")
    #     x_flattened_preprocessed = torch.from_numpy(preprocess_fn(flattened_images.numpy(), **preprocess_kwargs)).to(torch.float32)
    #     unflattened_images = unflatten_images_3d(x_flattened_preprocessed, n_channels, height, width)
    #     return DataLoader(
    #         unflattened_images,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         pin_memory=torch.cuda.is_available(),
    #     )

    def fit(self, dataset: IDataset, preprocess_fn=normalize_features):

        n_channels, width, height = dataset.image_shape
        assert width == height, "Error, need square input images."

        self.setup_device_and_seed()
        self.generator = self.generator.to(self.device)
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()

        optimizer, scheduler = self.setup_optimizer_and_scheduler()
        data_loader = self.setup_data_loader(dataset, preprocess_fn=preprocess_fn)
        loss_function = MMDLossConstrained(penalty=self.penalty, kernel=MixtureRQLinear())

        total_training_time = 0.0
        snapshot_duration = 0.0
        snapshot_intervals = [int(0.1 * i * self.epochs) for i in range(1, 11)]

        for epoch in range(self.epochs):
            print(f'\rEpoch {epoch} of {self.epochs}')
            generator_loss = 0
            mmd_loss_avg = 0
            epoch_start = time.time()

            for batch, embeddings in tqdm(data_loader, leave=False):
                batch = batch.to(self.device, non_blocking=True)
                embeddings = embeddings.to(self.device, non_blocking=True)
                optimizer.zero_grad()

                noise = torch.randn(batch.size(0), *self.generator.noise_dim, device=self.device)
                u_mappings = self.generator.sample_subspace_masks(noise)

                processed_batch = self.apply_subspaces_operator(batch, u_mappings)

                #embedded_batch = self.encode(batch)
                embedded_processed_batch = self.encode(processed_batch)

                batch_loss, mmd_loss = loss_function(embeddings, embedded_processed_batch, u_mappings)
                batch_loss.backward()
                optimizer.step()

                generator_loss += batch_loss.item() / len(data_loader)
                mmd_loss_avg += mmd_loss.item() / len(data_loader)

            epoch_duration = time.time() - epoch_start
            total_training_time += epoch_duration
            self.train_history["training_time"] = total_training_time

            if epoch in snapshot_intervals:
                self.train_history["training_time"] -= snapshot_duration
                snapshot_start = time.time()
                self.notify_logging_subscriber(dataset, epoch)
                snapshot_duration = time.time() - snapshot_start

            scheduler.step()
            print(f"Average loss in the epoch: {generator_loss}")
            self.train_history["generator_loss"].append(generator_loss)
            self.train_history["mmd_loss"].append(mmd_loss_avg)

        self.notify_logging_subscriber(dataset, self.epochs)
        self.train_history["training_time"] = total_training_time


    def __setup_data_loader(self, x_unflattened, cuda, mps):
        if cuda:
            return DataLoader(x_unflattened, batch_size=self.batch_size, drop_last=True, pin_memory=False, shuffle=True)
        else:  # Uses CUDA if available, otherwise MPS or nothing
            return DataLoader(x_unflattened, batch_size=self.batch_size, drop_last=True, pin_memory=mps, shuffle=True)

    def _generate_subspaces(self, count):

        generator_input_shape = self.generator.noise_dim

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

    def _create_mask_frequency_plot(self, count):
        u = self.sample_count_subspaces(count)
        u_agg = u.sum(dim=0)
        return u_agg / u.shape[0]

    def add_logger_subscriber(self, subscriber):
        self.logger_subscriber.append(subscriber)

    def notify_logging_subscriber(self, data: IDataset, epoch):
        for logger in self.logger_subscriber:
            logger.log(data, epoch)