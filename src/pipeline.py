import datetime
import os
from pathlib import Path

import torch

from src.models.Autoencoder import ResNet50AutoEncoder, ResNet18AutoEncoder
from src.models.ScaledAutoEncoder import ScaledAutoEncoder
from src.vmmd.VMMDFlattened import VMMDFlattened
from src.vmmd.VMMDLinearMapping import VMMDLinearMapping
from src.vmmd.VMMDRotationMapping import VMMDRotationMapping
from src.vmmd.VMMDSingleMask import VMMDSingleMask


class Pipeline:
    def __init__(self, lrs: [float], n_epochs=200, batch_size=32):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lrs = lrs

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.unscaled_resnet50 = ResNet50AutoEncoder()
        self.unscaled_resnet18 = ResNet18AutoEncoder()

        self.scaled_resnet50 = ScaledAutoEncoder(
            "../experiments/autoencoder/resnet50_2024-11-17 19:02:26.938377_scaled_autoencoder/train_history/resnet50_scaled_autoencoder_runs_100.pth")
        self.scaled_resnet18 = ScaledAutoEncoder(
            "../experiments/autoencoder/resnet18_2024-11-18 13:24:29.469347_scaled_autoencoder/train_history/resnet18_scaled_autoencoder_runs_100.pth")

    def  run(self, X):
        for lr in self.lrs:
            vmmd_flattened_unscaled_resnet50 = VMMDFlattened(epochs=self.n_epochs, path_to_directory=Path(
                os.getcwd()).parent / "experiments" / f"unscaled_resnet50_{datetime.datetime.now()}_{lr}_{self.n_epochs}",
                                                             lr=lr, batch_size=self.batch_size)

            vmmd_flattened_scaled_resnet50 = VMMDFlattened(epochs=self.n_epochs,
                                                           path_to_directory=Path(os.getcwd()).parent / "experiments" /
                                                                              f"scaled_resnet50_{datetime.datetime.now()}_{lr}_{self.n_epochs}",
                                                           lr=lr, batch_size=self.batch_size)

            vmmd_flattened_single_masked_resnet18 = VMMDSingleMask(epochs=self.n_epochs,
                                                                   path_to_directory=Path(os.getcwd()).parent / "experiments" /
                                                                              f"single_masked_unscaled_resnet18_{datetime.datetime.now()}_{lr}_{self.n_epochs}",
                                                                   lr=lr, batch_size=self.batch_size)

            vmmd_flattened_single_masked_resnet50 = VMMDSingleMask(epochs=self.n_epochs,
                                                                   path_to_directory=Path(
                                                                        os.getcwd()).parent / "experiments" /
                                                                                      f"single_masked_unscaled_resnet50_{datetime.datetime.now()}_{lr}_{self.n_epochs}",
                                                                   lr=lr, batch_size=self.batch_size)


            vmmd_flattened_unscaled_resnet18 = VMMDFlattened(epochs=self.n_epochs, path_to_directory=Path(
                os.getcwd()).parent / "experiments" /
                                                                                                 f"unscaled_resnet18_{datetime.datetime.now()}_{lr}_{self.n_epochs}",
                                                             lr=lr, batch_size=self.batch_size)
            vmmd_flattened_scaled_resnet18 = VMMDFlattened(epochs=self.n_epochs,
                                                           path_to_directory=Path(os.getcwd()).parent / "experiments" /
                                                                              f"scaled_resnet18_{datetime.datetime.now()}_{lr}_{self.n_epochs}",
                                                           lr=lr, batch_size=self.batch_size)

            vmmd_mapping_flattened_unscaled_resnet18 = VMMDLinearMapping(epochs=self.n_epochs,
                                                           path_to_directory=Path(os.getcwd()).parent / "experiments" /
                                                                              f"linear_mapping_resnet18_{datetime.datetime.now()}_{lr}_{self.n_epochs}",
                                                           lr=lr, batch_size=self.batch_size)

            vmmd_rotation_flattened_unscaled_resnet18 = VMMDRotationMapping(epochs=self.n_epochs,
                                                           path_to_directory=Path(os.getcwd()).parent / "experiments" /
                                                                              f"rotation_resnet18_{datetime.datetime.now()}_{lr}_{self.n_epochs}",
                                                           lr=lr, batch_size=self.batch_size)

            print("--------------------------[unscaled_resnet50]--------------------------")
            # vmmd_flattened_unscaled_resnet50.fit(X=X, autoencoder=unscaled_resnet50)
            #vmmd_flattened_single_masked_resnet50.fit(X=X, autoencoder=self.unscaled_resnet50)

            print("--------------------------[scaled_resnet50]--------------------------")
            # vmmd_flattened_scaled_resnet50.fit(X=X, autoencoder=scaled_resnet50)

            print("----------------------[single_mask_unscaled_resnet18]----------------------")
            #vmmd_flattened_single_masked_resnet18.fit(X, autoencoder=self.unscaled_resnet18)

            print("--------------------------[unscaled_resnet18]--------------------------")
            #vmmd_flattened_unscaled_resnet18.fit(X=X, autoencoder=self.unscaled_resnet18)

            print("--------------------------[scaled_resnet18]--------------------------")
            #vmmd_flattened_scaled_resnet18.fit(X=X, autoencoder=self.scaled_resnet18)

            print("--------------------------[linear_mapping_unscaled_resnet18]--------------------------")
            #vmmd_mapping_flattened_unscaled_resnet18.fit(X=X, autoencoder=self.unscaled_resnet18)

            print("--------------------------[linear_mapping_unscaled_resnet18]--------------------------")
            vmmd_rotation_flattened_unscaled_resnet18.fit(X=X, autoencoder=self.unscaled_resnet18)