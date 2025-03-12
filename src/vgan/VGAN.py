from src.vmmd.MMDLossConstrained import MMDLossConstrained
import time

from typing import Union

import torch
from collections import defaultdict
from src.data.IDataset import IDataset

from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import os
import torch_two_sample as tts
import torch.nn.functional as F

from src.utils.preprocessing import normalize_features
from src.utils.logger.vmmd.IVMMDLogger import IVMMDLogger
from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d, unflatten_images_3d
#from src.vmmd.MMDLossConstrained import MMDLossConstrained, MMDLossSquareRootConstrained, RBF


class VGAN:
    '''
    V-MMD, a Subspace-Generative Moment Matching Network.

    Class for the method VMMD, the application of a GMMN to the problem of Subspace Generation. As a GMMN, no
    kernel learning is performed. The default values for the kernel are
    '''

    def __init__(self, filename, generator, detector, penalty = MMDLossNoPenalty(), batch_size=500, temperature=1, epochs=30, lr_G=0.007, lr_D=0.007, iternum_d=1, iternum_g=5, momentum=0.99, seed=777, weight_decay=0.04, path_to_directory= Path(os.getcwd()).parent / "experiments" / "local",):
        self.storage = locals()
        self.train_history = defaultdict(list)
        self.batch_size = batch_size
        self.temperature = temperature
        self.epochs = epochs
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.iternum_d = iternum_d
        self.iternum_g = iternum_g
        self.momentum = momentum
        self.seed = seed
        self.weight_decay = weight_decay
        self.path_to_directory = path_to_directory
        self.generator_optimizer = None
        self.__elm = False
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.seed = 777
        self.filename = filename
        self.generator = generator
        self.detector = detector
        self.penalty = penalty
        self.logger_subscriber: list[IVMMDLogger] = []

    def add_logger_subscriber(self, subscriber):
        self.logger_subscriber.append(subscriber)

    def notify_logging_subscriber(self, data: IDataset, epoch):
        for logger in self.logger_subscriber:
            logger.log(data, epoch)

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

        x_data = extract_and_flatten_images_dataset_3d(x_data).to("cpu")
        x_flattened_normalized = torch.from_numpy(normalize(x_data, axis=0)).to(torch.float32)
        x_sample = torch.Tensor(pd.DataFrame(x_data).sample(count).to_numpy()).to(self.device)

        u_subspaces = self.sample_count_subspaces(count)
        x_sample = x_sample.view(-1, n_channels, height, width)
        ux_sample = self.apply_subspaces_operator(x_sample, u_subspaces)

        x_sample_embedded = self.encode(x_sample)
        ux_sample_embedded = self.encode(ux_sample)

        if type(bandwidth) == float:
            bandwidth = [bandwidth]

        mmd_loss = MMDLossConstrained()
        mmd_loss.forward(x_sample_embedded, ux_sample_embedded, u_subspaces * 1)
        self.bandwidth = mmd_loss.bandwidth

        bw = self.bandwidth.item()
        print("Bw: ", bw)
        mmd = tts.MMDStatistic(count, count)
        _, distances = mmd(x_sample_embedded, ux_sample_embedded, alphas=[bw], ret_matrix=True)
        pval = mmd.pval(distances)
        results.append(pval)
        print("Count: ", count, "PVal: ", pval)
        # TODO wtf?
        results.append(pval)

        bandwidth.append("recommended bandwidth")
        return pd.DataFrame([results], columns=bandwidth, index=["p-val"])

    def sample_count_subspaces(self, count):
        return self._generate_subspaces(count)

    def encode(self, x):

        if x.flatten(1).shape[1] != 224*224*3:
            x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)

        return self.detector.encode(x)

    def __normalize(self, x, dim=1):
        return x.div(x.norm(2, dim=dim).expand_as(x))

    def __distance(self, x, y, dist):
        """
        Computes distance between corresponding points points in `x` and `y`
        using distance `dist`.
        """
        if dist == 'L2':
            return (x - y).pow(2).mean()
        elif dist == 'L1':
            return (x - y).abs().mean()
        elif dist == 'cos':
            x_n = self.__normalize(x)
            y_n = self.__normalize(y)
            return 2 - (x_n).mul(y_n).mean()
        else:
            assert dist == 'none', 'wtf ?'

    def __weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0)

    def approx_subspace_dist(self, subspace_count=500):
        u = self.sample_count_subspaces(subspace_count)
        unique_subspaces, proba = np.unique(np.array(u.detach().to('cpu')), axis=0, return_counts=True)
        self.subspaces = torch.Tensor(unique_subspaces)
        self.proba = torch.Tensor(proba / proba.sum())

    def apply_subspaces_operator(self, x_sample_unflattened: torch.Tensor, u_subspaces: torch.Tensor):
        return u_subspaces * x_sample_unflattened

    def setup_device_and_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        elif torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)

    def fit(self, dataset: IDataset, preprocess_fn=normalize_features):

        n_channels, width, height = dataset.image_shape
        assert width == height, "Error, need square input images."

        self.setup_device_and_seed()
        self.generator = self.generator.to(self.device)
        self.detector = self.detector.to(self.device)
        self.generator.apply(self.__weights_init)
        self.detector.apply(self.__weights_init)

        # GRADIENT CLIPPING TO ENSURE LOCALLY LIPSCHITZ
        clip_param = 0.01 #as in WGAN paper
        torch.nn.utils.clip_grad_norm(self.detector.get_trainable_parameters(), clip_param)

        gen_optimizer, det_optimizer = self.setup_optimizer()
        self.generator_optimizer = gen_optimizer.__class__.__name__
        self.detector_optimizer = det_optimizer.__class__.__name__

        data_loader = self.setup_data_loader(dataset, n_channels, height, width, preprocess_fn=preprocess_fn)
        #loss_function = MMDLossConstrained(penalty=self.penalty, kernel=RBF())
        loss_function = MMDLossConstrainedFixKernel()

        total_training_time = 0.0
        snapshot_duration = 0.0
        snapshot_intervals = [int(i * 0.10 * self.epochs) for i in range(1, 11)]

        # OPTIMIZATION STUFF
        one = torch.mps.Tensor([1])
        minusone = one * -1

        # BATCH LOOP#
        iternum_d = 1
        iternum_g = 1
        detector_loss = np.nan
        generator_loss = np.nan

        for epoch in range(self.epochs):
            print(f'\rEpoch {epoch} of {self.epochs}')
            epoch_start = time.time()

            if iternum_g <= self.iternum_g:
                generator_loss = 0
                for batch in tqdm(data_loader, leave=False):
                    batch = batch.to(self.device, non_blocking=True)
                    self.batch_size = batch.size(0)
                    batch = batch.view(self.batch_size, -1)

                    noise = torch.randn(batch.size(0), *self.generator.noise_dim, device=self.device)
                    fake_subspaces = self.generator.sample_subspace_masks(noise) # Unfreeze G
                    fake_subspaces = fake_subspaces.view(self.batch_size, -1)
                    #fake_subspaces.requires_grad = True

                    projected_batch = fake_subspaces * batch

                    # GET SUBSPACES AND ENCODING-DECODING
                    batch = batch.view(self.batch_size, n_channels, height, width)
                    batch = F.interpolate(batch, size=224, mode='bilinear', align_corners=False)
                    batch_enc, _ = self.detector(batch)
                    batch_enc = batch_enc.view(self.batch_size, -1)

                    projected_batch = projected_batch.view(self.batch_size, n_channels, height, width)
                    projected_batch = F.interpolate(projected_batch, size=224, mode='bilinear', align_corners=False)
                    projected_batch_enc, _ = self.detector(projected_batch)
                    projected_batch_enc = projected_batch_enc.view(self.batch_size, -1)

                    # OPTIMIZATION STEP GENERATOR
                    self.detector.freeze_detector()

                    gen_optimizer.zero_grad()
                    total_batch_loss_G, mmd_loss = loss_function(batch_enc, projected_batch_enc, fake_subspaces)
                    self.bandwidth = loss_function.bandwidth
                    total_batch_loss_G.backward()

                    gen_optimizer.step()
                    generator_loss += float(total_batch_loss_G.to('cpu').detach().numpy()) / len(data_loader)

                iternum_g += 1
                if iternum_g > self.iternum_g:
                    iternum_d = 1

            elif iternum_d <= self.iternum_d:
                detector_loss = 0
                for batch in tqdm(data_loader, leave=False):
                    batch = batch.to(self.device, non_blocking=True)

                    batch = batch.view(batch.size(0), -1)
                    self.batch_size = batch.size(0)

                    # GET SUBSPACES AND ENCODING-DECODING
                    self.detector.unfreeze_decoder()

                    with torch.no_grad():
                        noise = torch.randn(batch.size(0), *self.generator.noise_dim, device=self.device)
                        fake_subspaces = self.generator.sample_subspace_masks(noise).clone().detach()
                        fake_subspaces = fake_subspaces.view(self.batch_size, -1)

                    projected_batch = self.apply_subspaces_operator(fake_subspaces, batch)

                    batch = batch.view(self.batch_size, n_channels, height, width)
                    batch = F.interpolate(batch, size=224, mode='bilinear', align_corners=False)
                    #batch = F.interpolate(batch, size=224, mode='nearest')

                    batch_enc, batch_dec = self.detector(batch)
                    batch = batch.view(self.batch_size, -1)
                    batch_dec = batch_dec.view(self.batch_size, -1)
                    batch_enc = batch_enc.view(self.batch_size, -1)

                    projected_batch = projected_batch.view(self.batch_size, n_channels, height, width)
                    projected_batch = F.interpolate(projected_batch, size=224, mode='bilinear', align_corners=False)
                    #projected_batch = F.interpolate(projected_batch, size=224, mode='nearest')

                    projected_batch_enc, projected_batch_dec = self.detector(projected_batch)
                    projected_batch = projected_batch.view(self.batch_size, -1)
                    projected_batch_enc = projected_batch_enc.view(self.batch_size, -1)
                    projected_batch_dec = projected_batch_dec.view(self.batch_size, -1)

                    L2_distance_batch = self.__distance(batch, batch_dec, 'L2')
                    L2_distance_projected_batch = self.__distance(projected_batch, projected_batch_dec, 'L2')

                    # OPTIMIZATION STEP DETECTOR
                    det_optimizer.zero_grad()
                    total_loss, mmd_loss = loss_function(batch_enc, projected_batch_enc, fake_subspaces)
                    batch_loss_D = minusone.to(self.device) * (total_loss - 0.1 * L2_distance_batch - 0.1 * L2_distance_projected_batch)  # Constrained MMD Loss

                    self.bandwidth = loss_function.bandwidth

                    batch_loss_D.backward()
                    det_optimizer.step()

                    detector_loss += float(batch_loss_D.to('cpu').detach().numpy()) / len(data_loader)

                iternum_d += 1

                if iternum_d > self.iternum_d:
                    iternum_g = 1

            epoch_duration = time.time() - epoch_start
            total_training_time += epoch_duration
            self.train_history["training_time"] = total_training_time

            # INBETWEEN SNAPSHOTS
            if epoch in snapshot_intervals:
                self.train_history["training_time"] -= snapshot_duration
                snapshot_start = time.time()
                self.notify_logging_subscriber(dataset, epoch)
                snapshot_duration = time.time() - snapshot_start

            print(f"Average loss in the epoch Generator: {generator_loss}")
            print(f"Average loss in the epoch Detector: {detector_loss}")
            self.train_history["generator_loss"].append(generator_loss)
            self.train_history["detector_loss"].append(detector_loss)

        self.notify_logging_subscriber(dataset, self.epochs)
        self.train_history["training_time"] = total_training_time

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

    def setup_optimizer(self):
        gen_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr_G, weight_decay=self.weight_decay, betas=(0.5, 0.999))

        det_optimizer = torch.optim.Adam(
            self.detector.get_trainable_parameters(), lr=self.lr_D, weight_decay=self.weight_decay, betas=(0.5, 0.999))
        return gen_optimizer, det_optimizer

    def setup_data_loader(self, dataset, n_channels, height, width, preprocess_fn=normalize_features,
                          **preprocess_kwargs):
        flattened_images = extract_and_flatten_images_dataset_3d(dataset).to("cpu")
        x_flattened_preprocessed = torch.from_numpy(preprocess_fn(flattened_images.numpy(), **preprocess_kwargs)).to(torch.float32)
        unflattened_images = unflatten_images_3d(x_flattened_preprocessed, n_channels, height, width)
        return DataLoader(
            unflattened_images,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )