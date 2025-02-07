import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from src.models.encoder.IdentityEncoder import IdentityEncoder
from src.vmmd import VMMD
from src.vmmd.logger.SubspaceDistributionPlotter import SubspaceDistributionPlotter
from src.vmmd.logger.SubspaceProjectionPlotter import SubspaceProjectionPlotter
from src.vmmd.logger.TrainingLogger import TrainingLogger
from src.models.encoder.pretrained_autoencoder.AutoEncoderManager import AutoEncoderManager



class VMMDWrapper:

    def __init__(self, vmmd: VMMD):
        self.vmmd = vmmd
        base_dir = self.__init_directory_paths()
        self.vmmd.add_logger_subscriber(TrainingLogger(vmmd, base_dir=base_dir))
        self.vmmd.add_logger_subscriber(SubspaceDistributionPlotter(vmmd, base_dir=base_dir))
        self.vmmd.add_logger_subscriber(SubspaceProjectionPlotter(vmmd, base_dir=base_dir))

    def load_model(self, path_to_generator_params: str):
        generator, autoencoder = self.__extract_models_from_file(path_to_generator_params)
        return self.vmmd.load_model(generator, autoencoder)

    def __extract_models_from_file(self, path_to_generator_params):
        pt_file_path = Path(path_to_generator_params)
        csv_file_path = pt_file_path.parent.parent / 'params.csv'

        filename = path_to_generator_params.split('/')[-1]
        train_iteration_number = int(re.search(r'\d+', filename).group())

        df = pd.read_csv(csv_file_path)

        noise_dim_column = df.loc[train_iteration_number, 'noise dim']
        img_shape_column = df.loc[train_iteration_number, 'image shape']
        autoencoder_column = df.loc[train_iteration_number, 'autoencoder']

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

        if not path_to_directory.exists():
            os.makedirs(path_to_directory, exist_ok=True)

        return path_to_directory

    def __extract_autoencoder(self, path_to_generator_params):
        pt_file_path = Path(path_to_generator_params)
        csv_file_path = pt_file_path.parent.parent / 'params.csv'
        first_row = pd.read_csv(csv_file_path, header=None).iloc[0]

        autoencoder_name = first_row['pretrained_autoencoder']

        autoencoder_manager = AutoEncoderManager()
        return autoencoder_manager.get_autoencoder(autoencoder_name)