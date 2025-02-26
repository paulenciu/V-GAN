import torch
from collections import defaultdict
import torch_two_sample as tts
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import operator
import datetime
from torch.autograd import Variable
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import torch_two_sample as tts
import torch.nn.functional as F

from src.models.encoder.pretrained_autoencoder.AutoEncoderManager import AutoEncoderManager
from src.vmmd.logger.ILogger import ILogger
from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty
from src.models.generator.AbstractGenerator import AbstractGenerator
from src.utils.BigUBuilder import calculate_average_u
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d, unflatten_images_3d
from src.vmmd.MMDLossConstrained import MMDLossConstrained, MMDLossSquareRootConstrained, RBF



class VGAN:
    '''
    V-MMD, a Subspace-Generative Moment Matching Network.

    Class for the method VMMD, the application of a GMMN to the problem of Subspace Generation. As a GMMN, no
    kernel learning is performed. The default values for the kernel are
    '''

    def __init__(self, filename, batch_size=500, temperature=1, epochs=30, lr_G=0.007, lr_D=0.007, iternum_d=1, iternum_g=5, momentum=0.99, seed=777, weight_decay=0.04, path_to_directory= Path(os.getcwd()).parent / "experiments" / "local",):
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
        self.logger_subscriber: list[ILogger] = []

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
        x_flattened_normalized = torch.from_numpy(normalize(x_data, axis=1)).to(torch.float32)
        x_sample = torch.Tensor(pd.DataFrame(x_flattened_normalized).sample(count).to_numpy()).to(self.device)

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

    def __normalize(x, dim=1):
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

    def fit(self, dataset: IDataset, detector, generator: AbstractGenerator):
        n_channels, width, height = dataset.image_shape
        assert width == height, "Error, need square input images."

        flattened_images = extract_and_flatten_images_dataset_3d(dataset).to("cpu")
        x_flattened_normalized = torch.from_numpy(normalize(flattened_images, axis=1)).to(torch.float32)
        unflattened_images = unflatten_images_3d(x_flattened_normalized, n_channels, height, width)  # Keep on CPU

        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()

        torch.manual_seed(self.seed)
        if cuda:
            torch.cuda.manual_seed(self.seed)
        elif mps:
            torch.mps.manual_seed(self.seed)

        # MODEL INTIALIZATION#
        epochs = self.epochs
        train_size = x_flattened_normalized.shape[0]
        self.batch_size = min(self.batch_size, train_size)

        detector = detector.to(self.device)
        generator = generator.to(self.device)

        generator.apply(self.__weights_init)
        detector.apply(self.__weights_init)

        gen_optimizer = torch.optim.Adadelta(
            generator.parameters(), lr=self.lr_G, weight_decay=self.weight_decay)

        det_optimizer = torch.optim.Adadelta(
            detector.parameters(), lr=self.lr_D, weight_decay=self.weight_decay)

        self.generator_optimizer = gen_optimizer.__class__.__name__
        self.detector_optimizer = det_optimizer.__class__.__name__

        loss_function = MMDLossConstrained()

        # OPTIMIZATION STUFF
        one = torch.mps.Tensor([1])
        minusone = one * -1

        # DATA LOADER#
        data_loader = self.__setup_data_loader(unflattened_images, cuda, mps)
        batch_number = data_loader.__len__()

        # BATCH LOOP#
        iternum_d = 1
        iternum_g = 1
        detector_loss = np.nan
        generator_loss = np.nan

        snapshot_intervals = [int(i * 0.10 * epochs) for i in range(1, 11)]

        # GET NOISE TENSORS#
        # GET NOISE TENSORS#
        noise_tensor = self.__setup_noise_tensor(generator_input_shape=generator.noise_dim)

        for epoch in range(self.epochs):
            print(f'\rEpoch {epoch} of {self.epochs}')

            # ELM
            if self.__elm == True:
                for p in detector.get_encoder().parameters():
                    p.requires_grad = False
            if iternum_d <= self.iternum_d:
                detector_loss = 0
                for batch in tqdm(data_loader, leave=False):
                    # Make sure there is only 1 observation per row.
                    batch = batch.view(self.batch_size, -1)
                    if cuda:
                        batch = batch.cuda()
                    elif mps:
                        batch = batch.to(torch.float32).to(
                            torch.device('mps'))  # float64 not suported with mps

                    # GET SUBSPACES AND ENCODING-DECODING
                    for p in detector.get_decoder().parameters():
                        p.requires_grad = True

                    batch = batch.view(-1, n_channels, width, height)
                    batch = F.interpolate(batch, size=224, mode='bilinear', align_corners=False)
                    batch_enc, batch_dec = detector(batch)

                    with torch.no_grad():
                        noise_tensor = Variable(noise_tensor.normal_())
                        fake_subspaces = Variable(generator.sample_subspace_masks(noise_tensor).clone().detach())  # Freeze G
                        fake_subspaces = F.interpolate(fake_subspaces, size=224, mode='bilinear', align_corners=False)

                    projected_batch_enc, projected_batch_dec = detector(fake_subspaces*batch)

                    L2_distance_batch = self.__distance(batch.view(self.batch_size, -1), batch_dec, 'L2')

                    L2_distance_projected_batch = self.__distance((fake_subspaces*batch).view(self.batch_size, -1), projected_batch_dec, 'L2')

                    # OPTIMIZATION STEP DETECTOR
                    det_optimizer.zero_grad()
                    batch_loss_D = minusone.to('mps')*(loss_function(batch_enc, projected_batch_enc, fake_subspaces) - .1 *
                                                       L2_distance_batch - .1*L2_distance_projected_batch)  # Constrained MMD Loss
                    self.bandwidth = loss_function.bandwidth
                    batch_loss_D.backward()
                    det_optimizer.step()
                    detector_loss += float(batch_loss_D.to(
                        'cpu').detach().numpy())/batch_number
                iternum_d += 1
                iternum_g = 1

            elif iternum_g <= self.iternum_g:
                generator_loss = 0
                for batch in tqdm(data_loader, leave=False):
                    # Make sure there is only 1 observation per row.
                    batch = batch.view(self.batch_size, -1)
                    if cuda:
                        batch = batch.cuda()
                    elif mps:
                        batch = batch.to(torch.float32).to(
                            torch.device('mps'))  # float64 not suported with mps
                    # GET SUBSPACES AND ENCODING-DECODING
                    batch_enc, batch_dec = detector(batch)
                    noise_tensor = Variable(noise_tensor.normal_())
                    fake_subspaces = Variable(
                        generator(noise_tensor))  # Unfreeze G
                    fake_subspaces.requires_grad = True
                    projected_batch_enc, projected_batch_dec = detector(
                        fake_subspaces*batch + torch.less(batch, 1/batch.shape[1])*torch.mean(batch, dim=0))
                    L2_distance_batch = self.__distance(
                        batch.view(self.batch_size, -1), batch_dec, 'L2')
                    L2_distance_projected_batch = self.__distance((fake_subspaces*batch + torch.less(
                        batch, 1/batch.shape[1])*torch.mean(batch, dim=0)).view(self.batch_size, -1), projected_batch_dec, 'L2')

                    # OPTIMIZATION STEP GENERATOR
                    for p in detector.parameters():
                        p.requires_grad = False
                    gen_optimizer.zero_grad()

                    batch_loss_G = loss_function(
                        batch_enc, projected_batch_enc, fake_subspaces)  # Constrained MMD Loss
                    self.bandwidth = loss_function.bandwidth
                    batch_loss_G.backward()
                    gen_optimizer.step()
                    generator_loss += float(batch_loss_G.to(
                        'cpu').detach().numpy())/batch_number
                iternum_g += 1
                if iternum_g > self.iternum_g:
                    iternum_d = 1
            # INBETWEEN SNAPSHOTS
            if epoch in snapshot_intervals:
                self.train_history["training_time"] -= snapshot_duration
                snapshot_start = time.time()
                self.notify_logging_subscriber(dataset)
                snapshot_duration = time.time() - snapshot_start

            print(f"Average loss in the epoch Generator: {generator_loss}")
            print(f"Average loss in the epoch Detector: {detector_loss}")
            self.train_history["generator_loss"].append(generator_loss)
            self.train_history["detector_loss"].append(detector_loss)

        if not self.path_to_directory == None:
            path_to_directory = Path(self.path_to_directory)
            if operator.not_(path_to_directory.exists()):
                os.mkdir(path_to_directory)
                if operator.not_(Path(path_to_directory/'models').exists()):
                    os.mkdir(path_to_directory / 'models')
            run_number = int(len(os.listdir(path_to_directory/'models'))/2)
            torch.save(generator.state_dict(),
                       path_to_directory/'models'/f'generator_{run_number}.pt')
            torch.save(generator.state_dict(),
                       path_to_directory/'models'/f'detector_{run_number}.pt')
            self.model_snapshot(path_to_directory, run_number, show=True)

        self.generator = generator
        self.detector = detector

    def _generate_subspaces(self, count):
        generator_input_shape = self.generator.noise_dim
        noise_tensor = self.__setup_noise_tensor(batch_size=count, generator_input_shape=generator_input_shape)

        if not self.seed is None:
            torch.manual_seed(self.seed)

        noise_tensor.normal_()
        u: torch.Tensor = self.generator.sample_subspace_masks(noise_tensor.to(self.device), mode="test")
        return u


    def __setup_data_loader(self, x_unflattened, cuda, mps):
        if cuda:
            return DataLoader(x_unflattened, batch_size=self.batch_size, drop_last=True, pin_memory=False, shuffle=True)
        else:  # Uses CUDA if available, otherwise MPS or nothing
            return DataLoader(x_unflattened, batch_size=self.batch_size, drop_last=True, pin_memory=mps, shuffle=True)


    def __setup_noise_tensor(self, generator_input_shape: torch.Tensor, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        shape = (batch_size, *generator_input_shape)
        return torch.empty(shape, dtype=torch.float32, device=self.device)