import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union

import torch
from collections import defaultdict

from sklearn.preprocessing import normalize
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import operator
import torch_two_sample as tts

from src.models.Mmd_loss_constrained import MMDLossConstrained, RBF
from src.models.autoencoder.AutoEncoderManager import AutoEncoderManager
from src.models.generator.AbstractGenerator import AbstractGenerator
from src.models.generator.diagonal_matrix.three_channels.GeneratorThreeChannel import GeneratorThreeChannel
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannel import GeneratorOneChannel
from src.models.autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
from src.utils.BigUBuilder import create_big_u
from src.utils.ImageFlattenerUtility import flatten_images_dataset_3d, unflatten_images_3d

def tensor_to_image(tensor):
    """
    Converts a PyTorch tensor to a numpy image array.
    Ensures that the dtype is compatible with matplotlib.
    """
    tensor = tensor.to(torch.float32)  # Ensure float32 type
    array = tensor.cpu().numpy()  # Convert to numpy
    array = np.clip(array, 0, 1)  # Ensure values are in [0, 1] range
    return array.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

class VMMDRef(ABC):
    """
       V-MMD, a Subspace-Generative Moment Matching Network.

       Class for the method VMMD, the application of a GMMN to the problem of Subspace Generation. As a GMMN, no
       kernel learning is performed. The default values for the kernel are
    """

    def __init__(self, filename, batch_size=500, epochs=30, lr=0.007, momentum=0.99, seed=777, weight_decay=0.04,
                 path_to_directory= Path(os.getcwd()).parent / "experiments" / "local", flattened_projection=False):
        self.autoencoder = None
        self.generator = None
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
        self.flattened_projection = flattened_projection #states if the mask produced are flattened into a single vector when applied to image
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    @abstractmethod
    def sample_count_subspaces(self, count):
        """
        Samples count many subspaces of shape (batch_size, 3, 32, 32).
        """
        pass

    def get_params(self) -> dict:
        return {'batch size': self.batch_size, 'epochs': self.epochs, 'lr_g': self.lr,
                'momentum': self.momentum, 'weight decay': self.weight_decay,
                'batch_size': self.batch_size, 'seed': self.seed,
                'generator optimizer': self.generator_optimizer,
                'generator name': self.generator.__class__.__name__,
                'noise dim': self.generator.noise_dim,
                'autoencoder': self.autoencoder.__class__.__name__}

    def load_model(self, path_to_generator_params: str):
        generator, autoencoder = self.__extract_models_from_file(path_to_generator_params)
        self.generator = generator.to(self.device)
        self.autoencoder = autoencoder.to(self.device)
        self.generator.load_state_dict(torch.load(path_to_generator_params))
        self.generator.eval()
        self.generator_optimizer = f'Loaded Model from {path_to_generator_params} with {generator.noise_dim} dimensions in the latent space'



    def __extract_models_from_file(self, path_to_generator_params):
        pt_file_path = Path(path_to_generator_params)
        csv_file_path = pt_file_path.parent.parent / 'params.csv'

        filename = path_to_generator_params.split('/')[-1]
        train_iteration_number = int(re.search(r'\d+', filename).group())

        df = pd.read_csv(csv_file_path)

        noise_dim_column = df.loc[train_iteration_number, 'noise dim']
        noise_tensor = eval(str(noise_dim_column).replace('tensor', 'torch.tensor'))

        generator_name = df.loc[train_iteration_number, "generator name"] + "(noise_tensor)"
        generator = eval(generator_name)

        autoencoder_name = df.loc[train_iteration_number, "autoencoder"] + "()"
        autoencoder = eval(autoencoder_name)

        return generator, autoencoder

    def __plot_loss(self, path_to_directory, data, encoder, show=False, run_number=0):

        myopic_test_df = self.check_if_myopic(data, encoder)
        pval_recommended_bw = myopic_test_df.iat[0, 1]

        train_history = self.train_history
        plt.style.use('ggplot')
        generator_y = train_history['generator_loss']
        x = np.linspace(1, len(generator_y), len(generator_y))
        fig, ax = plt.subplots()

        ax.plot(x, generator_y, color="cornflowerblue",
                label="Generator loss", linewidth=2)
        ax.plot([], [], ' ', label="pval: " + str(pval_recommended_bw))
        ax.plot([], [], ' ', label="generator: " + self.generator.__class__.__name__)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        ax.legend(loc="upper right")
        plt.savefig(path_to_directory / f"train_history_{run_number}.pdf",
                    format="pdf", dpi=1200)

        if show == True:
            print("The show option has been depricated due to lack of utility")

    def check_if_myopic(self, x_data, emb_func, bandwidth: Union[float, np.array] = 0.01, count=500) -> pd.DataFrame:
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

        x_data = flatten_images_dataset_3d(x_data).to("cpu")

        x_data = normalize(x_data, axis=0)
        x_sample = torch.Tensor(pd.DataFrame(
            x_data).sample(count).to_numpy()).to(self.device)

        u_subspaces = self.sample_count_subspaces(count)

        x_sample = x_sample.view(-1, 3, 32, 32)

        ux_sample = self.apply_subspaces_operator(x_sample, u_subspaces)

        x_sample_embedded = emb_func(x_sample).squeeze()
        ux_sample_embedded = emb_func(ux_sample).squeeze()

        if type(bandwidth) == float:
            bandwidth = [bandwidth]

        if not hasattr(self, 'bandwidth'):
            mmd_loss = MMDLossConstrained(0, flattened=self.flattened_projection)
            mmd_loss.forward(
                x_sample_embedded, ux_sample_embedded, u_subspaces * 1)
            self.bandwidth = mmd_loss.bandwidth

        bw = self.bandwidth.item()
        mmd = tts.MMDStatistic(count, count)
        _, distances = mmd(x_sample_embedded, ux_sample_embedded, alphas=[bw], ret_matrix=True)
        pval = mmd.pval(distances)
        results.append(pval)
        print("Count: ", count, "PVal: ", pval)
        results.append(pval)

        bandwidth.append("recommended bandwidth")
        return pd.DataFrame([results], columns=bandwidth, index=["p-val"])

    def apply_subspaces_operator(self, x_sample_unflattened: torch.Tensor, u_subspaces: torch.Tensor):
        return u_subspaces * x_sample_unflattened

    def fit(self, X_dataset, autoencoder, generator: AbstractGenerator):

        n_channels, width, height = X_dataset[0][0].shape[0], X_dataset[0][0].shape[1], X_dataset[0][0].shape[2]
        assert width == height, "Error, need square input images."

        self.autoencoder = autoencoder
        encoder = autoencoder.get_encoder().to(self.device)

        X_flattened = flatten_images_dataset_3d(X_dataset).to(self.device)
        X_unflattened = unflatten_images_3d(X_flattened, 3, 32, 32).to(self.device)

        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()

        torch.manual_seed(self.seed)
        if cuda:
            torch.cuda.manual_seed(self.seed)
        elif mps:
            torch.mps.manual_seed(self.seed)

        # MODEL INTIALIZATION#
        epochs = self.epochs
        train_size = X_flattened.shape[0]
        self.batch_size = min(self.batch_size, train_size)

        # SETUP GENERATOR OPTIMIZATION
        self.generator = generator.to(self.device)
        optimizer = torch.optim.Adadelta(
            self.generator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.generator_optimizer = optimizer.__class__.__name__

        loss_function = MMDLossConstrained(weight=10, kernel=RBF(), flattened=self.flattened_projection)

        snapshot_intervals = [int(i * 0.25 * epochs) for i in range(1, 5)]

        #INITIAL SNAPSHOT
        self.__store_model_snapshot(X_dataset, encoder)

        for epoch in range(epochs):
            print(f'\rEpoch {epoch} of {epochs}')
            generator_loss = 0

            # DATA LOADER#
            data_loader = self.__setup_data_loader(X_unflattened, cuda, mps)
            batch_number = data_loader.__len__()

            # GET NOISE TENSORS#
            noise_tensor = self.__setup_noise_tensor(generator_input_shape=generator.noise_dim, mps=mps, cuda=cuda)

            # BATCH LOOP#
            for batch in tqdm(data_loader, leave=False):
                # Make sure there is only 1 observation per row.
                #batch = batch.view(self.batch_size, -1)
                if cuda:
                    batch = batch.cuda()
                elif mps:
                    batch = batch.to(torch.float32).to(
                        torch.device('mps'))  # float64 not suported with mps

                # SAMPLE NOISE#
                noise_tensor.normal_()

                # OPTIMIZATION STEP#
                optimizer.zero_grad()
                u_mappings = generator.sample_subspace_masks(noise_tensor).to(self.device)

                processed_batch = self.apply_subspaces_operator(batch, u_mappings)

                embedded_batch = encoder(batch).squeeze()
                embedded_fake_subspaces = encoder(processed_batch).squeeze()

                batch_loss = loss_function(embedded_batch, embedded_fake_subspaces, u_mappings)

                self.bandwidth = loss_function.bandwidth
                batch_loss.backward()
                optimizer.step()
                generator_loss += float(batch_loss.to(
                    'cpu').detach().numpy()) / batch_number

            #INBETWEEN SNAPSHOTS
            if epoch in snapshot_intervals:
                self.__store_model_snapshot(X_dataset, encoder)
            print(f"Average loss in the epoch: {generator_loss}")
            self.train_history["generator_loss"].append(generator_loss)

        #FINAL SNAPSHOT
        self.__store_model_snapshot(X_dataset, encoder)

        self.generator = generator

    def __model_snapshot(self, X, encoder, path_to_directory=None, run_number=0, show=False):
        ''' Creates an snapshot of the model

        Saves important information regarding the training of the model
        Args:
            - path_to_directory (Path): Specifies the path to directory (relative to the WD)
            - show (bool): Boolean specifying if a pop-up window should open to show the plot for previsualization.
        '''

        if path_to_directory == None:
            path_to_directory = self.path_to_directory
        path_to_directory = Path(path_to_directory)
        if not path_to_directory.exists():
            os.mkdir(path_to_directory)

        self.__include_train_history(path_to_directory, X, encoder, run_number, show)
        self.__include_visual_plots(X=X, n_samples=5, n_masks=5, path_to_experiment=path_to_directory, run_number=run_number)

    def __include_train_history(self, path_to_directory, X, encoder, run_number=0, show=False):
        path_to_train_history = path_to_directory / "train_history"

        if not path_to_train_history.exists():
            os.mkdir(path_to_train_history)

        path_to_train_history_csv = path_to_directory / "csv"
        path_to_train_history_plot = path_to_directory / "train_history"

        if not path_to_train_history_csv.exists():
            os.mkdir(path_to_train_history_csv)

        if not path_to_train_history_plot.exists():
            os.mkdir(path_to_train_history_plot)

        pd.DataFrame(self.train_history["generator_loss"]).to_csv(
            path_to_train_history_csv / f'generator_loss_{run_number}.csv', header=False, index=False)

        if os.path.isfile(path_to_directory / 'params.csv') != True:
            pd.DataFrame([self.get_params()], [0]).to_csv(
                path_to_directory / 'params.csv')
        else:
            params = pd.read_csv(path_to_directory / 'params.csv', index_col=0)
            params_new = pd.DataFrame([self.get_params()], [run_number])
            params = params.reindex(params.index.union(params_new.index))
            params.update(params_new)
            params.to_csv(
                path_to_directory / 'params.csv')
        self.__plot_loss(path_to_directory=path_to_train_history_plot, show=show, run_number=run_number, data=X, encoder=encoder)

    def __setup_data_loader(self, X_unflattened, cuda, mps):
        if cuda:
            return DataLoader(
                X_unflattened, batch_size=self.batch_size, drop_last=True, pin_memory=False, shuffle=True)
        else:  # Uses CUDA if Available, other wise MPS or nothing
            return DataLoader(
                X_unflattened, batch_size=self.batch_size, drop_last=True, pin_memory=mps, shuffle=True)

    def __store_model_snapshot(self, X, encoder):
        path_to_directory = Path(self.path_to_directory)
        current_date = datetime.now().strftime("%d-%m")

        path_to_directory = path_to_directory / current_date / self.filename

        if not path_to_directory.exists():
            os.makedirs(path_to_directory, exist_ok=True)

        models_path = Path(path_to_directory / 'models')

        if not models_path.exists():
            os.makedirs(models_path)

        run_number = int(len(os.listdir(models_path)))

        if not models_path.exists():
            os.mkdir(models_path)

        torch.save(self.generator.state_dict(),
                   path_to_directory / 'models' / f'generator_{run_number}.pt')
        self.__model_snapshot(path_to_directory=path_to_directory, run_number=run_number, X=X, encoder=encoder, show=True)

    def _generate_subspaces(self, count, generate_subspace_adjust=True):

        generator_input_shape = self.generator.noise_dim

        # Need to load in cpu as mps Tensor module doesn't properly fix the seed
        noise_tensor = self.__setup_noise_tensor(batch_size=count, generator_input_shape=generator_input_shape, mps=False, cuda=False)

        if not self.seed is None:
            torch.manual_seed(self.seed)

        noise_tensor.normal_()
        u = self.generator.sample_subspace_masks(noise_tensor.to(self.device))

        if generate_subspace_adjust:
            u = torch.greater_equal(u, 1 / u.shape[1])
        return u

    def __setup_noise_tensor(self, generator_input_shape: torch.Tensor, mps: bool, cuda: bool, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        # Determine tensor shape based on generator_input_shape
        if  generator_input_shape.shape == torch.Size([1]):
            shape = (batch_size, generator_input_shape[0])
        elif generator_input_shape.shape == torch.Size([3]):
            shape = (batch_size, generator_input_shape[0], generator_input_shape[1], generator_input_shape[2])
        else:
            raise NotImplementedError

        # Select appropriate device and tensor type
        device = 'cuda' if cuda else 'mps' if mps else 'cpu'
        return torch.empty(shape, dtype=torch.float32, device=device)



    def __include_visual_plots(self, X, n_samples, n_masks, path_to_experiment, run_number):
        sample_indices = np.arange(n_samples)
        X_sample = torch.utils.data.Subset(X, sample_indices)

        fig, axis = plt.subplots(n_samples + 1, 2 + n_masks, figsize=(5 * (2 + n_masks), 5 * (n_samples + 1)))
        u = self.sample_count_subspaces(n_masks).to(self.device).detach()

        big_u, _, _ = create_big_u(u)
        big_u = big_u.to(torch.float32).to(self.device)

        axis[0, 0].imshow(tensor_to_image(torch.ones(3, 32, 32)))
        axis[0, 0].axis("off")

        for i in range(n_masks):
            axis[0, i + 1].imshow(tensor_to_image(u[i]))
            axis[0, i + 1].axis("off")
            axis[0, i + 1].set_title(f"Mask {i + 1}")


        axis[0, n_masks + 1].imshow(tensor_to_image(big_u))
        axis[0, n_masks + 1].axis("off")
        axis[0, n_masks + 1].set_title(f"Big U Mask")

        for i in range(1, n_samples + 1):

            image, _ = X_sample[i - 1]
            image = image.to(torch.float32).to(self.device)

            axis[i, 0].imshow(tensor_to_image(image))
            axis[i, 0].set_title(f"Original {i + 1}")
            axis[i, 0].axis("off")

            u = self.sample_count_subspaces(n_masks).to(self.device)
            image = image.unsqueeze(0).repeat(n_masks, 1, 1, 1).to(self.device)
            ux_data = self.apply_subspaces_operator(image, u).to(self.device).detach()


            for j in range(n_masks):
                axis[i, j + 1].imshow(tensor_to_image(ux_data[j]))
                axis[i, j + 1].set_title(f"Projection {j + 1}")
                axis[i, j + 1].axis("off")

            big_u_image = self.apply_subspaces_operator(u_subspaces=big_u, x_sample_unflattened=image[0].squeeze())
            big_u_image = big_u_image.to(torch.float32).to(self.device)

            axis[i, n_masks + 1].imshow(tensor_to_image(big_u_image))
            axis[i, n_masks + 1].axis("off")
            axis[i, n_masks + 1].set_title(f"Big U Projection")


        plt.tight_layout()
        plt.show()

        if path_to_experiment is not None:
            fig.tight_layout()
            path_to_directory = Path(path_to_experiment)
            if not Path(path_to_directory / 'plots').exists():
                os.mkdir(path_to_directory / 'plots')

            fig.savefig(path_to_directory / 'plots' / f'plot_{run_number}.png')

    def __extract_noise_dim(self, train_iteration_number, path_to_generator_params) -> torch.Tensor:
        pt_file_path = Path(path_to_generator_params)
        csv_file_path = pt_file_path.parent.parent / 'params.csv'

        df = pd.read_csv(csv_file_path, header=None)
        row = df.iloc[train_iteration_number]

        noise_dim_column = row['noise dim']

        # Convert to torch.Tensor
        return eval(str(noise_dim_column).replace('tensor', 'torch.tensor'))

    def __extract_autoencoder(self, path_to_generator_params):
        pt_file_path = Path(path_to_generator_params)
        csv_file_path = pt_file_path.parent.parent / 'params.csv'
        first_row = pd.read_csv(csv_file_path, header=None).iloc[0]

        autoencoder_name = first_row['autoencoder']

        autoencoder_manager = AutoEncoderManager()
        return autoencoder_manager.get_autoencoder(autoencoder_name)



