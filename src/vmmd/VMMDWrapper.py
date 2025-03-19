import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from src.models.encoder.IdentityEncoder import IdentityEncoder
from src.utils.logger.vmmd.GradiantPlotter import GradiantPlotter
from src.vmmd import VMMD
from src.utils.logger.vmmd.SubspaceDistributionPlotter import SubspaceDistributionPlotter
from src.utils.logger.vmmd.SubspaceProjectionPlotter import SubspaceProjectionPlotter
from src.utils.logger.vmmd.TrainingLogger import TrainingLogger
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4Softmax import GeneratorOneChannelV4Softmax


class VMMDWrapper:

    def __init__(self, vmmd: VMMD):
        self.vmmd = vmmd
        base_dir = self.__init_directory_paths()
        self.vmmd.add_logger_subscriber(TrainingLogger(vmmd, base_dir=base_dir))
        #self.vmmd.add_logger_subscriber(GradiantPlotter(vmmd))
        self.vmmd.add_logger_subscriber(SubspaceDistributionPlotter(vmmd, base_dir=base_dir))
        self.vmmd.add_logger_subscriber(SubspaceProjectionPlotter(vmmd, base_dir=base_dir))

    def load_model(self, path_to_generator_params: str):
        generator, autoencoder = self.__extract_models_from_file(path_to_generator_params)
        len_model_dir = len("/models/generator_x.pt")
        self.vmmd.path_to_directory = path_to_generator_params[:-len_model_dir]
        return self.vmmd.load_model(generator, autoencoder)

    def get_path_to_directory(self, path_to_generator_params):
        return "/".join(path_to_generator_params.split('/')[:-3])

    def get_run_number_from_generator_path(self, path_to_generator_params: str):
        filename = path_to_generator_params.split('/')[-1]
        train_iteration_number = int(re.search(r'\d+', filename).group())
        return train_iteration_number

    def __extract_models_from_file(self, path_to_generator_params):
        pt_file_path = Path(path_to_generator_params)
        csv_file_path = pt_file_path.parent.parent / 'params.csv'

        train_iteration_number = self.get_run_number_from_generator_path(path_to_generator_params)

        df = pd.read_csv(csv_file_path)

        noise_dim_column = df.loc[train_iteration_number, 'noise dim']
        img_shape_column = df.loc[train_iteration_number, 'image shape']
        autoencoder_column = df.loc[train_iteration_number, 'pretrained_autoencoder']

        noise_tensor = eval(str(noise_dim_column).replace('tensor', 'torch.tensor'))
        img_shape = eval(img_shape_column)

        generator_name = df.loc[train_iteration_number, "generator name"] + "(noise_tensor, img_shape)"
        generator = eval(generator_name)


        if autoencoder_column == "NoneType":
            autoencoder = IdentityEncoder()
        else:
            autoencoder_name = autoencoder_column + "()"
            autoencoder = eval(autoencoder_name)

        generator.load_state_dict(torch.load(path_to_generator_params, map_location=torch.device('cpu')))
        generator.generator_optimizer = f'Loaded Model from {path_to_generator_params} with {generator.noise_dim} dimensions in the latent space'
        return generator, autoencoder


    def __init_directory_paths(self):
        path_to_directory = Path(self.vmmd.path_to_directory)
        if not path_to_directory.exists():
            os.mkdir(path_to_directory)

        current_date = datetime.now().strftime("%d-%m")

        path_to_directory = path_to_directory / current_date / self.vmmd.filename
        self.vmmd.path_to_directory = path_to_directory

        if not path_to_directory.exists():
            os.makedirs(path_to_directory, exist_ok=True)

        return path_to_directory
