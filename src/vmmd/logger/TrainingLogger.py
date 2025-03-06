import os
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

from src.data import IDataset
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d
from src.vgan.VGAN import VGAN
from src.vmmd.VMMD import VMMD
from src.vmmd.MMDLossConstrained import MMDLossConstrained
from src.vmmd.logger.ILogger import ILogger


class TrainingLogger(ILogger):

    def __init__(self, vmmd: VMMD, base_dir: Path):
        self.vmmd = vmmd
        self.base_dir = base_dir
        (self.path_to_train_history_csv,
         self.path_to_train_history_plot,
         self.path_to_model_params) = self.__init_base_dir(base_dir)

    def log(self, data, epochs=0):
        run_number = int(len(os.listdir(self.path_to_model_params)))

        self.__update_model_parms(run_number)
        self.__update_generator_loss(run_number)
        self.__update_params(run_number)
        self.__plot_loss(run_number, data)
        #self.__plot_mmd_val()

    def __plot_mmd_val(self):
        train_history = self.vmmd.train_history
        plt.style.use('ggplot')
        XX = train_history['XX']
        XY = train_history['XY']
        YY = train_history['YY']
        bw = train_history["bandwidth"]

        x = np.linspace(1, len(XX), len(XX))
        fig, ax = plt.subplots()

        plt.plot(x, XX, color="cornflowerblue", linewidth=2)
        plt.title("XX")
        plt.show()

        plt.plot(x, YY, color="red", label="YY", linewidth=2)
        plt.title("YY")
        plt.show()

        plt.plot(x, XY, color="green", label="XY", linewidth=2)
        plt.title("XY")
        plt.show()

        plt.plot(x, bw, color="orange", label="bw", linewidth=2)
        plt.title("BW")

        plt.show()


        #ax.legend(loc="upper right")
        #plt.show()

    def __init_base_dir(self, base_dir):
        path_to_train_history_csv = base_dir / "csv"
        path_to_train_history_plot = base_dir / "train_history"
        path_to_model_params = base_dir / "models"

        if not path_to_model_params.exists():
            os.mkdir(path_to_model_params)

        if not path_to_train_history_csv.exists():
            os.mkdir(path_to_train_history_csv)

        if not path_to_train_history_plot.exists():
            os.mkdir(path_to_train_history_plot)

        return path_to_train_history_csv, path_to_train_history_plot, path_to_model_params

    def get_params(self) -> dict:
        vmmd = self.vmmd

        if isinstance(vmmd, VMMD):
            return {'batch size': vmmd.batch_size, 'epochs': vmmd.epochs, 'lr_g': vmmd.lr,
                    'momentum': vmmd.momentum, 'weight decay': vmmd.weight_decay,
                    'batch_size': vmmd.batch_size, 'seed': vmmd.seed,
                    'generator optimizer': vmmd.generator_optimizer,
                    'generator name': vmmd.generator.__class__.__name__,
                    'image shape': vmmd.generator.img_shape,
                    'noise dim': vmmd.generator.noise_dim,
                    'pretrained_autoencoder': vmmd.encoder.__class__.__name__,
                    'mmd_penalty': vmmd.penalty.__class__.__name__,
                    'mmd_penalty_stats': vmmd.penalty.get_stats()}
        elif isinstance(vmmd, VGAN):
            return {'batch size': vmmd.batch_size, 'epochs': vmmd.epochs, 'lr_g': vmmd.lr_G, 'lr_d': vmmd.lr_D,
                    'momentum': vmmd.momentum, 'weight decay': vmmd.weight_decay,
                    'batch_size': vmmd.batch_size, 'seed': vmmd.seed,
                    'generator optimizer': vmmd.generator_optimizer,
                    'generator name': vmmd.generator.__class__.__name__,
                    'image shape': vmmd.generator.img_shape,
                    'noise dim': vmmd.generator.noise_dim,
                    'autoencoder': vmmd.detector.__class__.__name__,
                    'autoencoder_latent_size': vmmd.detector.finetune_latent_dim,
                    'mmd_penalty': vmmd.penalty.__class__.__name__,
                    'mmd_penalty_stats': vmmd.penalty.get_stats()}

    def __update_generator_loss(self, run_number=0):
        pd.DataFrame(self.vmmd.train_history["generator_loss"]).to_csv(
            self.path_to_train_history_csv / f'generator_loss_{run_number}.csv', header=False, index=False)

    def __update_params(self, run_number=0):
        if os.path.isfile(self.base_dir / 'params.csv') is False:
            pd.DataFrame([self.get_params()], [0]).to_csv(self.base_dir / 'params.csv')

        else:
            params = pd.read_csv(self.base_dir / 'params.csv', index_col=0)
            params_new = pd.DataFrame([self.get_params()], [run_number])
            params = params.reindex(params.index.union(params_new.index))
            params.update(params_new)
            params.to_csv(self.base_dir / 'params.csv')

    def __plot_loss(self, run_number, data, sample_count=500):
        myopic_test_df = self.vmmd.check_if_myopic(data, count=sample_count)
        pval_of_recommended_bw = myopic_test_df.iat[0, 1]
        n_unique_subspaces = self.__count_unique_subspaces(count=sample_count)

        train_history = self.vmmd.train_history
        plt.style.use('ggplot')
        generator_y = train_history['generator_loss']
        x = np.linspace(1, len(generator_y), len(generator_y))
        fig, ax = plt.subplots()

        if isinstance(self.vmmd, VGAN):
            detector_y = train_history['detector_loss']
            ax.plot(x, detector_y, color="red", label="Detector loss", linewidth=2)

        else:
            mmd_y = train_history['mmd_loss']
            ax.plot(x, mmd_y, color="red", label="MMD loss", linewidth=2)

        training_time = train_history['training_time']
        ax.plot(x, generator_y, color="cornflowerblue", label="Generator loss", linewidth=2)

        ax.plot([], [], ' ', label="pval: " + str(pval_of_recommended_bw))
        ax.plot([], [], ' ', label="n_u_subs: " + str(n_unique_subspaces) + f"/{sample_count}")
        ax.plot([], [], ' ', label="generator: " + self.vmmd.generator.__class__.__name__)
        ax.plot([], [], ' ', label="tr_time: " + training_time.__str__())

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        ax.legend(loc="upper right")
        plt.savefig(self.path_to_train_history_plot / f"train_history_{run_number}.pdf", format="pdf", dpi=1200)
        plt.show()

    def __count_unique_subspaces(self, count):
        u = self.vmmd.sample_count_subspaces(count=count)
        unique_subspaces, _ = np.unique(np.array(u.detach().to('cpu')), axis=0, return_counts=True)
        return len(unique_subspaces)

    def __update_model_parms(self, run_number=0):
        torch.save(self.vmmd.generator.state_dict(),
                   self.path_to_model_params / f'generator_{run_number}.pt')