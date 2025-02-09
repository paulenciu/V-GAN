import re
import time

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union

import torch
from collections import defaultdict
from src.data.IDataset import IDataset

from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
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
from src.utils.BigUBuilder import create_big_u
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d, unflatten_images_3d
from src.vmmd.MMDLossConstrainedV2 import MMDLossConstrainedV2


class IndexedDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index], index

    def __len__(self):
        return len(self.tensor)


class VMMDM(ABC):
    def __init__(self, filename, batch_size=500, epochs=30, lr=0.007, momentum=0.99, seed=None, weight_decay=0.04,
                 path_to_directory=Path(os.getcwd()).parent / "experiments" / "local", penalty=MMDLossNoPenalty()):
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
        self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                   else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.logger_subscriber: list[ILogger] = []
        self.embedded_dataset = None

    def add_logger_subscriber(self, subscriber):
        self.logger_subscriber.append(subscriber)

    def notify_logging_subscriber(self, data: IDataset):
        for logger in self.logger_subscriber:
            logger.log(data)

    @abstractmethod
    def sample_count_subspaces(self, count):
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
        if count > len(x_data):
            count = len(x_data)
        results = []

        n_channels, height, width = x_data.image_shape

        x_data = extract_and_flatten_images_dataset_3d(x_data).to("cpu")
        indices = torch.randperm(len(x_data))[:count]
        x_sample = x_data[indices].to(self.device)
        x_sample = x_sample.view(-1, n_channels, height, width)

        u_subspaces = self.sample_count_subspaces(count)
        ux_sample = self.apply_subspaces_operator(x_sample, u_subspaces)

        if x_sample.shape[1] == 1:
            x_sample = x_sample.repeat(1, 3, 1, 1)
        if ux_sample.shape[1] == 1:
            ux_sample = ux_sample.repeat(1, 3, 1, 1)

        with torch.no_grad():
            x_sample_embedded = self.encode(x_sample)
            ux_sample_embedded = self.encode(ux_sample)

        if type(bandwidth) == float:
            bandwidth = [bandwidth]

        if not hasattr(self, 'bandwidth'):
            mmd_loss = MMDLossConstrainedV2()
            mmd_loss.forward(x_sample_embedded, ux_sample_embedded, u_subspaces * 1)
            self.bandwidth = mmd_loss.bandwidth

        bw = self.bandwidth.item()
        mmd = tts.MMDStatistic(count, count)
        _, distances = mmd(x_sample_embedded, ux_sample_embedded, alphas=[bw], ret_matrix=True)
        pval = mmd.pval(distances)
        results.append(pval)
        results.append(pval)

        bandwidth.append("recommended bandwidth")
        return pd.DataFrame([results], columns=bandwidth, index=["p-val"])

    def apply_subspaces_operator(self, x_sample_unflattened: torch.Tensor, u_subspaces: torch.Tensor):
        return u_subspaces * x_sample_unflattened

    def encode(self, x):
        assert x.shape[1] == 3, f"Encoder requires 3 channels, but got {x.shape[1]}."
        return self.encoder(x).view(x.shape[0], -1)

    def fit(self, dataset: IDataset, encoder, generator: AbstractGenerator):
        n_channels, width, height = dataset.image_shape
        assert width == height, "Input images must be square."

        flattened_images = extract_and_flatten_images_dataset_3d(dataset).to("cpu")
        unflattened_images = unflatten_images_3d(flattened_images, n_channels, height, width).to(self.device)

        self.encoder = encoder.to(self.device).eval()

        # Precompute embeddings for the entire dataset
        self._precompute_embeddings(unflattened_images)

        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()
        torch.manual_seed(self.seed)
        if cuda:
            torch.cuda.manual_seed(self.seed)
        elif mps:
            torch.mps.manual_seed(self.seed)

        self.generator = generator.to(self.device)
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.generator_optimizer = optimizer.__class__.__name__

        loss_function = MMDLossConstrainedV2(penalty=self.penalty, kernel=RBF())


        indexed_dataset = IndexedDataset(unflattened_images)
        data_loader = self.__setup_data_loader(indexed_dataset, cuda, mps)
        batch_number = len(data_loader)

        noise = self.__setup_noise_tensor(generator.noise_dim, mps, cuda)
        total_training_time = 0.0
        snapshot_duration = 0.0


        epochs = self.epochs
        snapshot_intervals = [int(i * 0.10 * epochs) for i in range(1, 11)]
        self.notify_logging_subscriber(dataset)

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            generator_loss = 0.0
            mmd_loss_avg = 0.0
            epoch_start = time.time()

            for batch, indices in tqdm(data_loader, leave=False):
                if cuda:
                    batch = batch.cuda(non_blocking=True)
                elif mps:
                    batch = batch.to(self.device, non_blocking=True)

                embedded_batch = self.embedded_dataset[indices].to(self.device, non_blocking=True)
                noise.normal_()

                optimizer.zero_grad()
                u_mappings = self.generator.sample_subspace_masks(noise)
                processed_batch = self.apply_subspaces_operator(batch, u_mappings)

                if processed_batch.shape[1] == 1:
                    processed_batch = processed_batch.repeat(1, 3, 1, 1)

                # Use gradient checkpointing for memory efficiency
                processed_batch.requires_grad_(True)
                embedded_processed_batch = torch.utils.checkpoint.checkpoint(self.encode, processed_batch)

                batch_loss, mmd_loss = loss_function(embedded_batch, embedded_processed_batch, u_mappings)
                self.bandwidth = loss_function.bandwidth

                batch_loss.backward()
                optimizer.step()

                generator_loss += batch_loss.item() / batch_number
                mmd_loss_avg += mmd_loss.item() / batch_number

                del u_mappings, processed_batch, embedded_processed_batch
                if cuda:
                    torch.cuda.empty_cache()

            epoch_duration = time.time() - epoch_start
            total_training_time += epoch_duration
            self.train_history["training_time"] = total_training_time
            self.train_history["generator_loss"].append(generator_loss)
            self.train_history["mmd_loss"].append(mmd_loss_avg)
            print(f'Epoch Loss: {generator_loss:.4f}')

            # INBETWEEN SNAPSHOTS
            if epoch in snapshot_intervals:
                self.train_history["training_time"] -= snapshot_duration
                snapshot_start = time.time()
                self.notify_logging_subscriber(dataset)
                snapshot_duration = time.time() - snapshot_start

        self.notify_logging_subscriber(dataset)

    def _precompute_embeddings(self, unflattened_images):
        self.encoder.eval()
        batch_size_encode = 512
        embedded_dataset = []

        with torch.no_grad():
            for i in range(0, len(unflattened_images), batch_size_encode):
                batch = unflattened_images[i:i + batch_size_encode]
                if batch.shape[1] == 1:
                    batch = batch.repeat(1, 3, 1, 1)
                embedded_batch = self.encode(batch.to(self.device)).cpu()
                embedded_dataset.append(embedded_batch)

        self.embedded_dataset = torch.cat(embedded_dataset, dim=0).cpu()

    def __setup_data_loader(self, x_unflattened, cuda, mps):
        if cuda:
            return DataLoader(x_unflattened, batch_size=self.batch_size, drop_last=True, pin_memory=False, shuffle=True)
        else:  # Uses CUDA if available, otherwise MPS or nothing
            return DataLoader(x_unflattened, batch_size=self.batch_size, drop_last=True, pin_memory=mps, shuffle=True)

    def _generate_subspaces(self, count, threshold=None):
        noise_tensor = self.__setup_noise_tensor(
            self.generator.noise_dim,
            mps=False,
            cuda=False,
            batch_size=count
        )
        if self.seed is not None:
            torch.manual_seed(self.seed)
        noise_tensor.normal_()
        u = self.generator.sample_subspace_masks(noise_tensor.to(self.device), mode="test")
        return u

    def __setup_noise_tensor(self, generator_input_shape, mps, cuda, batch_size=None):
        batch_size = batch_size or self.batch_size
        shape = (batch_size, *generator_input_shape)
        return torch.empty(shape, dtype=torch.float32, device="mps" if mps else "cuda" if cuda else "cpu")