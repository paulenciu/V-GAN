import time
from abc import ABC, abstractmethod
from typing import Union
import torch
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import GradScaler

from src.data.IDataset import IDataset
from sklearn.preprocessing import normalize
from src.data.dataset.PreEmbeddedDataset import PreEmbeddedDataset
from src.utils.preprocessing import normalize_features
from src.vmmd.VMMD import VMMD
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset
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


class VMMDEmbedding(VMMD):
    """
    V-MMD with embedding-space subspace operations and pixel-space MMD.
    """

    def __init__(self, filename, autoencoder, generator: AbstractGenerator,
                 batch_size=500, epochs=30, lr=0.007, momentum=0.99, seed=None,
                 weight_decay=0.04, path_to_directory=Path(os.getcwd()).parent / "experiments" / "local",
                 penalty=MMDLossNoPenalty(), kernel=RBF()):
        super().__init__(filename, autoencoder, generator, batch_size, epochs, lr, momentum, seed,
                         weight_decay, path_to_directory, penalty)
        self.decoder = autoencoder.get_decoder_and_freeze()
        self.encoder_input_shape = autoencoder.get_encoder_input_shape()
        self.decoder_input_shape = autoencoder.get_decoder_input_shape()
        self.scaler = GradScaler()
        self.kernel = kernel


    def sample_count_subspaces(self, count):
        return self._generate_subspaces(count)

    def _create_mask_frequency_plot(self, count):
        u = self.sample_count_subspaces(count)
        u_height = int(u.shape[1] / 224)
        u_width = int(u.shape[1] / u_height)

        u = u.view(-1, 1, u_width, u_height)
        u = u.repeat(1, 3, 1, 1)  # makes image black / white
        u_agg = u.sum(dim=0)
        return u_agg / u.shape[0]

    def check_if_myopic(self, x_data: IDataset, bandwidth: Union[float, np.array] = 0.01, count=500) -> pd.DataFrame:
        count = min(count, len(x_data))
        results = []
        n_channels, height, width = x_data.image_shape

        with torch.no_grad():
            x_sample = next(iter(DataLoader(x_data, batch_size=count)))[0].to(self.device)
            embeddings = self.encode(x_sample)

            noise = torch.randn(count, *self.generator.noise_dim, device=self.device)
            u_mappings = self.generator.sample_subspace_masks(noise)

            processed_embeddings = embeddings * u_mappings
            processed_images = processed_embeddings.view(x_sample.size(0), *self.decoder_input_shape)
            reconstructed_images = self.decoder(processed_images)

        if isinstance(bandwidth, float):
            bandwidth = [bandwidth]

        mmd_loss = MMDLossConstrained()
        x_sample = x_sample.view(x_sample.size(0), -1)
        reconstructed_images = reconstructed_images.view(reconstructed_images.size(0), -1)
        mmd_loss.forward(x_sample, reconstructed_images, u_mappings * 1)
        self.bandwidth = mmd_loss.bandwidth
        bw = self.bandwidth.item()
        print("Bw: ", bw)
        mmd = tts.MMDStatistic(count, count)

        _, distances = mmd(x_sample.view(x_sample.size(0), -1),
                           reconstructed_images.view(reconstructed_images.size(0), -1), alphas=[bw], ret_matrix=True)
        pval = mmd.pval(distances)
        results.append(pval)
        print("Count: ", count, "PVal: ", pval)
        # TODO wtf?
        results.append(pval)

        bandwidth.append("recommended bandwidth")
        return pd.DataFrame([results], columns=bandwidth, index=["p-val"])

    def encode(self, x):
        return self.encoder(x).view(x.shape[0], -1)

    def load_model(self, generator: AbstractGenerator, autoencoder):
        if autoencoder is not None:
            self.encoder = autoencoder.get_encoder_and_freeze().to(self.device)
            self.decoder = autoencoder.get_decoder_and_freeze().to(self.device)

        self.generator = generator.to(self.device)
        self.generator.eval()


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

    def fit(self, dataset: IDataset, preprocess_fn=normalize_features, set_encoder_eval=True):
        self.setup_device_and_seed()
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.generator = self.generator.to(self.device)

        if set_encoder_eval:
            self.encoder.eval()

        optimizer, _ = self.setup_optimizer_and_scheduler()
        data_loader = self.setup_data_loader(dataset, preprocess_fn=preprocess_fn)
        loss_function = MMDLossConstrained(penalty=self.penalty, kernel=self.kernel)
        total_training_time = 0.0
        snapshot_intervals = [int(0.1 * i * self.epochs) for i in range(1, 11)]


        for epoch in range(self.epochs):
            epoch_start = time.time()
            generator_loss = 0
            mmd_loss_avg = 0

            for images, embeddings in tqdm(data_loader, leave=False):
                images = images.to(self.device)
                embeddings = embeddings.to(self.device)
                optimizer.zero_grad()

                noise = torch.randn(images.size(0), *self.generator.noise_dim, device=self.device)
                u_mappings = self.generator.sample_subspace_masks(noise)

                processed_images = self.apply_subspaces_operator(embeddings, u_mappings, is_embedding=True)

                reconstructed = torch.utils.checkpoint.checkpoint(
                    self._decode_with_memory, processed_images
                )
                #
                # processed_images = processed_images.view(images.size(0), *self.decoder_input_shape)
                # reconstructed = self.decoder(processed_images)

                images = images.view(images.size(0), -1)
                reconstructed = reconstructed.view(images.size(0), -1)

                # with torch.no_grad():
                flat_images = images.view(images.size(0), -1)
                flat_recon = reconstructed.view(reconstructed.size(0), -1)
                batch_loss, mmd_loss = loss_function(flat_images, flat_recon, u_mappings)
                #batch_loss, mmd_loss = loss_function(images, reconstructed, u_mappings)

                total_norm = 0.0
                for p in self.generator.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                batch_loss.backward()
                optimizer.step()

                # self.scaler.scale(batch_loss).backward()
                # self.scaler.step(optimizer)
                # self.scaler.update()

                del images, embeddings, noise, u_mappings, reconstructed

                # batch_loss.backward()
                # optimizer.step()

                generator_loss += batch_loss.item() / len(data_loader)
                mmd_loss_avg += mmd_loss.item() / len(data_loader)

            epoch_duration = time.time() - epoch_start
            total_training_time += epoch_duration
            self.train_history["training_time"] = total_training_time

            if epoch in snapshot_intervals:
                self.notify_logging_subscriber(dataset, epoch)

            self.train_history["generator_loss"].append(generator_loss)
            self.train_history["mmd_loss"].append(mmd_loss_avg)
            print(f"Epoch {epoch + 1}/{self.epochs} | Loss: {generator_loss:.4f}")

        self.notify_logging_subscriber(dataset, self.epochs)

    def _decode_with_memory(self, processed):
        """Helper method for checkpointing"""
        return self.decoder(processed.view(processed.size(0), *self.decoder_input_shape))

    def apply_subspaces_operator(self, x_sample: torch.Tensor, u_subspaces: torch.Tensor, is_embedding=False,
                                 output_image_size=64):
        """:param is_embedding: If False, the input is encoded-decoded"""
        u_subspaces = u_subspaces.to(torch.float32)

        if not is_embedding:

            # Encoder expects 4D input
            if len(x_sample.shape) != 4:
                x_sample = x_sample.unsqueeze(0)

            x_sample = self.encode(x_sample)

        projection = x_sample * u_subspaces.view(*x_sample.shape)
        return projection if is_embedding else interpolate(
            self.decoder(projection.view(projection.size(0), *self.decoder_input_shape)), size=output_image_size,
            mode="bilinear")