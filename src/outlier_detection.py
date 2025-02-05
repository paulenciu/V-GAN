import random
import re
import time
from typing import Any

import numpy as np
import torch
import torchvision
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD
from pyod.models.lof import LOF
from pyod.models.feature_bagging import FeatureBagging
from pathlib import Path
import datetime
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score, f1_score
from sel_suod.models.base import sel_SUOD
import itertools
from sklearn.preprocessing import label_binarize
from joblib.externals.loky import get_reusable_executor
import json
import os
import logging

from src.data.IDataset import IDataset
from src.data.dataset_loader import load_data
from src.data.dataset_type import DatasetType
from src.models.autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
from src.models.generator.AbstractGenerator import AbstractGenerator
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV3 import GeneratorOneChannelV3
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4 import GeneratorOneChannelV4
from src.utils.ImageFlattenerUtility import flatten_images_dataset_3d
from src.vmmd.VMMDDiagonal1Channel import VMMDDiagonal1Channel
from src.vmmd.penalty.MMDLossPenalty import MMDLossDiscretePenalty, MMDLossL2Penalty, MMDLossPenaltyJoin

logger = logging.getLogger(__name__)

def aggregator_funct(decision_function: np.array, type: str = "avg", weights: np.ndarray = None) -> np.ndarray:
    assert type in ["avg", "exact"], f"{type} aggregation not found"

    if type == "avg":
        return np.average(decision_function, axis=1, weights=weights)

    if type == "exact":
        weights = weights/weights.sum()
        random_indexes = random.choices(
            range(decision_function.shape[1]), k=decision_function.shape[0])
        aggregated_scores = [weights[random_indexes[i]]
                             * (decision_function[i])[random_indexes[i]] for i in range(decision_function.shape[0])]
        return aggregated_scores


def launch_outlier_detection_experiments(autoencoder, generator: AbstractGenerator, dataset_type: DatasetType, category: list[str],
                                         base_estimators: list, epochs: int = 10, seed: int = 777, image_size=(224, 224),
                                         subspace_count=100, path_to_directory=None, store_stats=True) -> dict:
    """Launch the outlier detection experiments for a given data

    Args:
    Returns:
        tuple: Returns the AUC, PRAUC and F1 of the ensemble obtained by VGAN subspaces
    """
    logger.info("No instance of a pretrained generation model found. Proceeding to train a new Generator.")
    X_test, X_train, Y_test = __prepare_data(category, dataset_type, image_size)

    vgan = VMMDDiagonal1Channel(epochs=epochs, seed=seed, path_to_directory=path_to_directory)
    vgan.fit(dataset=X_train, autoencoder=autoencoder, generator=generator)
    decision_function_scores_ens, decision_time, fit_time = __launch_outlier_detection_ensemble(X_test, X_train,
                                                                                                base_estimators, seed,
                                                                                                subspace_count, vgan)
    stats = __calculate_occ_stats(Y_test, dataset_type, decision_function_scores_ens, decision_time, fit_time)
    path_to_store = Path(vgan.path_to_directory) / "od_stats.csv"
    return stats if not store_stats else __store_stats_csv(path_to_store, stats)


def __store_stats_csv(path, stats: dict) -> None:
    stats = pd.DataFrame([stats])
    path= Path(path)
    stats.to_csv(path, index=False)

def __prepare_data(category, dataset_type, image_size):
    X_train, X_test, Y_test = load_data(dataset_type=dataset_type, category=category, image_size=image_size)
    X_train = flatten_images_dataset_3d(X_train).cpu().numpy()
    X_test = flatten_images_dataset_3d(X_test).cpu().numpy()
    return X_train, X_test, Y_test


def pretrained_launch_outlier_detection_experiments(path_to_generator: str, dataset_type: DatasetType, category: list[str],
                                                    base_estimators: list, seed: int = 777, subspace_count=100,
                                                    image_size=(224, 224), store_stats=True) -> dict | None:
    """Launch the outlier detection experiments for a given data

    Args:
        dataset_name (str): Name of the data to load
        base_estimators (list): List including all base estimators to build the ensemble. If the length is = 1, 
        then an homogeneus ensemble will be fitted.
        directory (Path): Path to the directory one wishes to load to
    Returns:
        np.array: Returns the AUC, PRAUC and F1
    """
    logger.info(
        f"Pretrained generator found!")
    x_train, x_test, y_test = __prepare_data(dataset_type=dataset_type, category=category, image_size=image_size)

    vgan = VMMDDiagonal1Channel()
    vgan.load_model(path_to_generator)

    decision_function_scores_ens, decision_time, fit_time = __launch_outlier_detection_ensemble(x_test, x_train,
                                                                                                base_estimators, seed,
                                                                                                subspace_count, vgan)

    iteration_number = re.search(r'\d+', path_to_generator.split("/")[-1]).group()
    path_to_folder = "/".join([p for p in path_to_generator.split("/")[:-2]])
    filename = "od_stats_" + iteration_number + ".csv"
    path = Path(path_to_folder) / filename

    stats = __calculate_occ_stats(y_test, dataset_type, decision_function_scores_ens, decision_time, fit_time)
    return stats if not store_stats else __store_stats_csv(path, stats)


def __launch_outlier_detection_ensemble(x_test, x_train, base_estimators, seed, subspace_count, vgan):

    vgan.seed = seed
    vgan.approx_subspace_dist(add_leftover_features=False, subspace_count=subspace_count)
    subspaces = vgan.subspaces
    print("Number of unique subspaces:", len(subspaces), "/", subspace_count)

    ensemble_model = sel_SUOD(base_estimators=base_estimators, subspaces=subspaces,
                              n_jobs=-1, bps_flag=False, approx_flag_global=False)

    fit_time_start = time.time()
    ensemble_model.fit(x_train)
    fit_time = time.time() - fit_time_start

    decision_time_start = time.time()
    decision_function_scores_ens = ensemble_model.decision_function(x_test)
    decision_time = time.time() - decision_time_start

    decision_function_scores_ens = aggregator_funct(
        decision_function_scores_ens, weights=vgan.proba, type="avg")

    return decision_function_scores_ens, decision_time, fit_time


def __calculate_occ_stats(y_test, dataset_type, decision_function_scores_ens, decision_time, fit_time) -> dict:
    return {"Dataset": dataset_type,
            "AUC": auc(y_test, decision_function_scores_ens),
            "PRAUC": average_precision_score(y_test, decision_function_scores_ens),
            "F1": f1_score(y_test, (decision_function_scores_ens > np.quantile(decision_function_scores_ens, .80)) * 1),
            "Training Time": str(datetime.timedelta(seconds=fit_time)),
            "Decision Time": str(datetime.timedelta(seconds=decision_time))}


def check_if_myopicity_was_uphold(dataset_name: str, gen_model_to_use="VGAN") -> tuple:
    """Given the data name, the function will return the p-value of the GOF test using the MMD with the recommended
    bandwidth in [Look for the paper I don't remember the surname of the authors rn].

    Args:
        dataset_name (str): Name of the data as included in the json file 'datasets_file_name.json'

    Returns:
        float: p-value for the two-sampe non-parametric GoF test using the MMD with recommended bandwidth (by L2 distances)
    """
    X_train, _, _ = load_data(dataset_name)

    if gen_model_to_use == "VGAN":
        vgan = VGAN()
        vgan.load_models(Path() / "experiments" / "VGAN" /
                         f"VGAN_{dataset_name}" / "models" / "generator_0.pt", ndims=X_train.shape[1])
    elif gen_model_to_use == "VMMD":
        vgan = VMMD()
        vgan.load_models(Path() / "experiments" / "VMMD" /
                         f"VMMD_{dataset_name}" / "models" / "generator_0.pt", ndims=X_train.shape[1])
    vgan.approx_subspace_dist()

    return vgan.check_if_myopic(X_train, bandwidth=[
        1, 0.1, 0.001, 0.0001], count=min(1000, X_train.shape[0]))["recommended bandwidth"].item(), vgan.subspaces.shape[0]

def launch_outlier_detection_baseline(dataset_type: DatasetType, category: list[str], base_estimator, image_size=(224, 224), root_dir="../experiments/baselines"):

    X_train, X_test, Y_test = load_data(dataset_type=dataset_type, category=category, image_size=image_size)

    X_train = flatten_images_dataset_3d(X_train).cpu().numpy()
    X_test = flatten_images_dataset_3d(X_test).cpu().numpy()

    fit_time_start = time.time()
    base_estimator.fit(X_train)
    fit_time = time.time() - fit_time_start

    decision_time_start = time.time()
    decision_function_scores = base_estimator.decision_function(X_test)
    decision_time = time.time() - decision_time_start

    stats = pd.DataFrame([__calculate_occ_stats(Y_test, dataset_type, decision_function_scores, decision_time, fit_time)])

    path_to_dir = Path(root_dir) / str(dataset_type.name)
    filename = base_estimator.__class__.__name__ + str(image_size[0]) + ".csv"
    os.makedirs(path_to_dir, exist_ok=True)
    stats.to_csv(path_to_dir / filename, index=False)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dataset_name = "Ionosphere"

    auc_vgan_ens = pretrained_launch_outlier_detection_experiments(dataset_name, [
        LOF()], gen_model_to_use="VMMD")  # ,   epochs=3000, temperature=1)
    print(
        f'AUC obtained by the VGAN-based ensemble model: {print(auc_vgan_ens)}')