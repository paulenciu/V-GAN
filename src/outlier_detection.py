import random

import numpy as np
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
from src.models.autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV3 import GeneratorOneChannelV3
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4 import GeneratorOneChannelV4
from src.utils.ImageFlattenerUtility import flatten_images_dataset_3d
from src.vmmdref.VMMDDiagonal1Channel import VMMDDiagonal1Channel
from src.vmmdref.penalty.MMDLossPenalty import MMDLossDiscretePenalty, MMDLossL2Penalty, MMDLossPenaltyJoin

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


def launch_outlier_detection_experiments(dataset_name: str, category: list[str],
                                         base_estimators: list, epochs: int = 10,
                                         temperature: float = 10, seed: int = 777, gen_model_to_use: str = "VMMD") -> dict:
    """Launch the outlier detection experiments for a given data

    Args:
        dataset_name (str): Name of the data to load
        base_estimators (list): List including all base estimators to build the ensemble. If the length is = 1, 
        then an homogeneus ensemble will be fitted.
        directory (Path): Path to the directory one wishes to load to
    Returns:
        tuple: Returns the AUC, PRAUC and F1 of the ensemble obtained by VGAN subspaces
    """
    logger.info(
        "No instance of a pretrained generation model found. Proceeding to train a new Generator.")
    X_train, X_test, Y_test= load_data(dataset_name=dataset_name, category=category, image_size=(32, 32))

    vgan = VMMDDiagonal1Channel()
    vgan.load_model("../experiments/local/28-01/od_mvtec_1D-1/models/generator_0.pt")
    #vgan = VMMDDiagonal1Channel()
    noise_dim = 128

    #vgan.fit(X_train, ResNet18AutoEncoder(), GeneratorOneChannelV4(noise_dim, X_train.image_shape))

    vgan.seed = seed
    subspaces = vgan.sample_count_subspaces(500)
    subspaces_numpy = subspaces.reshape(subspaces.shape[0], -1).cpu().numpy()
    ensemble_model = sel_SUOD(base_estimators=base_estimators, subspaces=subspaces_numpy,
                              n_jobs=-1, bps_flag=False, approx_flag_global=False)

    X_train = flatten_images_dataset_3d(X_train).cpu().numpy()
    X_test = flatten_images_dataset_3d(X_test).cpu().numpy()

    ensemble_model.fit(X_train)

    decision_function_scores_ens = ensemble_model.decision_function(X_test)
    decision_function_scores_ens = aggregator_funct(
        decision_function_scores_ens, weights=vgan.proba, type="avg")
    return {"Dataset": dataset_name,
            "AUC": auc(Y_test, decision_function_scores_ens),
            "PRAUC": average_precision_score(Y_test, decision_function_scores_ens),
            "F1": f1_score(Y_test, (decision_function_scores_ens > np.quantile(decision_function_scores_ens, .80)) * 1)}


def pretrained_launch_outlier_detection_experiments(dataset_name: str, base_estimators: list, seed: int = 777, gen_model_to_use: str = "VGAN") -> tuple:
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
    X_train, X_test, y_test = load_data(dataset_name)

    if gen_model_to_use == "VGAN":
        logger.debug("Loading VGAN")
        vgan = VGAN()
        vgan.load_models(Path() / "experiments" / "VGAN" /
                         f"VGAN_{dataset_name}" / "models" / "generator_0.pt", ndims=X_train.shape[1])
    elif gen_model_to_use == "VMMD":
        logger.debug("Loading VMMD")
        vgan = VMMD()
        vgan.load_models(Path() / "experiments" / "VMMD" /
                         f"VMMD_{dataset_name}" / "models" / "generator_0.pt", ndims=X_train.shape[1])
    vgan.seed = seed
    vgan.approx_subspace_dist(add_leftover_features=False)
    ensemble_model = sel_SUOD(base_estimators=base_estimators, subspaces=vgan.subspaces,
                              n_jobs=-1, bps_flag=False, approx_flag_global=False)
    ensemble_model.fit(X_train)
    decision_function_scores_ens = ensemble_model.decision_function(
        X_test)
    decision_function_scores_ens = aggregator_funct(
        decision_function_scores_ens, weights=vgan.proba, type="avg")

    print(vgan.subspaces.shape[0])
    return {"Dataset": dataset_name,
            "AUC": auc(y_test, decision_function_scores_ens),
            "PRAUC": average_precision_score(y_test, decision_function_scores_ens),
            "F1": f1_score(y_test, (decision_function_scores_ens > np.quantile(decision_function_scores_ens, .80)) * 1)}


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dataset_name = "Ionosphere"

    auc_vgan_ens = pretrained_launch_outlier_detection_experiments(dataset_name, [
        LOF()], gen_model_to_use="VMMD")  # ,   epochs=3000, temperature=1)
    print(
        f'AUC obtained by the VGAN-based ensemble model: {print(auc_vgan_ens)}')