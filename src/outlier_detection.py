import random
import time

import numpy as np
from pathlib import Path
import datetime
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score, f1_score
from sel_suod.models.base import sel_SUOD
import os
import logging

from sklearn.preprocessing import normalize
from src.data.dataset_loader import load_data
from src.data.dataset_type import DatasetType
from src.models.encoder.AbstractEncoder import AbstractEncoder
from src.models.generator.AbstractGenerator import AbstractGenerator
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d, unflatten_images_3d
from src.vmmd.VMMDWrapper import VMMDWrapper
from src.vmmd.model.VMMDDiagonal1Channel import VMMDDiagonal1Channel
from src.vmmd.outlier_detection.VMMDOD import VMMDOD
from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty

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


def launch_outlier_detection_experiments(filename: str, encoder: AbstractEncoder, generator: AbstractGenerator, dataset_type: DatasetType,
                                         category: list[str], base_estimators: list, epochs: int = 10, lr=0.5, seed: int = 777,
                                         image_size=(224, 224), subspace_count=100, path_to_directory=None, store_stats=True,
                                         penalty=MMDLossNoPenalty(), batch_size=256, momentum=0.8, weight_decay = 0.1,
                                         standardize_data=False, skip_od=False) -> dict:
    """Launch the outlier detection experiments for a given data

    Args:
    Returns:
        tuple: Returns the AUC, PRAUC and F1 of the ensemble obtained by VGAN subspaces
    """
    logger.info("No instance of a pretrained generation model found. Proceeding to train a new Generator.")

    x_train = load_data(dataset_type=dataset_type, category=category, image_size=image_size, standardize=standardize_data)

    vmmd = VMMDDiagonal1Channel(epochs=epochs, seed=seed, path_to_directory=path_to_directory,
                                lr=lr, penalty=penalty, filename=filename,
                                batch_size=batch_size, momentum=momentum, weight_decay=weight_decay)

    vmmd_wrapper = VMMDWrapper(vmmd)

    vmmd.fit_memory_efficient(dataset=x_train, encoder=encoder, generator=generator)

    if skip_od:
        return None

    vmmd_od = VMMDOD(vmmd=vmmd)

    decision_function_scores_ens, decision_time, fit_time, unique_subspace_count, y_test = __launch_outlier_detection_ensemble(x_train,
                                                                                                                               base_estimators, seed,
                                                                                                                               subspace_count, vmmd,
                                                                                                                               dataset_type=dataset_type,
                                                                                                                               category=category,
                                                                                                                               image_size=image_size,
                                                                                                                               standardize=standardize_data)

    stats = __calculate_occ_stats(y_test, dataset_type, decision_function_scores_ens, decision_time, fit_time)
    return stats if not store_stats else vmmd_od.store_od_stats(stats, run_number=-1)


def pretrained_launch_outlier_detection_experiments(path_to_generator: str, dataset_type: DatasetType, category: list[str],
                                                    base_estimators: list, seed: int = 777, subspace_count=100,
                                                    image_size=(224, 224), store_stats=True, standardize_data=False) -> dict | None:
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
    x_train = load_data(dataset_type=dataset_type, category=category, image_size=image_size, standardize=standardize_data, train=True)

    vmmd = VMMDDiagonal1Channel()

    vmmd_wrapper = VMMDWrapper(vmmd)
    vmmd_wrapper.load_model(path_to_generator)

    vmmd.path_to_directory = vmmd_wrapper.get_path_to_directory(path_to_generator)

    predictions, decision_time, fit_time, unique_subspace_count, y_test = __launch_outlier_detection_ensemble(x_train,
                                                                                                           base_estimators, seed,
                                                                                                           subspace_count, vmmd,
                                                                                                           dataset_type=dataset_type,
                                                                                                           category=category,
                                                                                                           image_size=image_size,
                                                                                                           standardize=standardize_data)

    vmmd_od = VMMDOD(vmmd=vmmd)

    stats = __calculate_occ_stats(y_test, dataset_type, predictions, decision_time, fit_time)
    train_iteration_number = vmmd_wrapper.get_run_number_from_generator_path(path_to_generator)
    return stats if not store_stats else vmmd_od.store_od_stats(stats, train_iteration_number)


def __launch_outlier_detection_ensemble(x_train, base_estimators, seed, sample_subspace_count, vgan, dataset_type, category, image_size, standardize, subspace_distance_lambda=0.5, classifier_delta=1):

    vgan.seed = seed
    vgan.approx_subspace_dist(subspace_count=sample_subspace_count)
    subspaces = np.array(vgan.subspaces , dtype=int)
    unique_subspace_count = len(subspaces)
    print("Number of unique subspaces:", unique_subspace_count, "/", sample_subspace_count)

    ensemble_model = sel_SUOD(base_estimators=base_estimators, subspaces=subspaces,
                              n_jobs=4, bps_flag=False, approx_flag_global=False)

    x_train_flattened = extract_and_flatten_images_dataset_3d(x_train).to("cpu")
    x_train_flattened = normalize(x_train_flattened, axis=0)

    fit_time_start = time.time()
    ensemble_model.fit(x_train_flattened)
    fit_time = time.time() - fit_time_start

    # not needed anymore
    del x_train, x_train_flattened

    x_test, y_test = load_data(dataset_type=dataset_type, category=category, image_size=image_size, standardize=standardize, train=False)

    x_test_flattened = extract_and_flatten_images_dataset_3d(x_test).to("cpu")
    x_test_flattened = normalize(x_test_flattened, axis=0)
    y_test = np.array(y_test)

    n_samples = x_test_flattened.shape[0]
    decision_function_scores_ens = np.zeros(n_samples)

    batch_size = 512
    decision_time_start = time.time()

    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = x_test_flattened[i:end_idx]

        batch_scores = ensemble_model.decision_function(batch)

        decision_function_scores_ens[i:end_idx] = aggregator_funct(
            batch_scores,
            weights=vgan.proba,
            type="avg"
        )

    decision_time = time.time() - decision_time_start

    subspace_min_distances = __calculate_min_subspace_distances(subspaces, x_test_flattened)
    dist_tensor = subspace_distance_lambda * subspace_min_distances

    predictions = classifier_delta * decision_function_scores_ens + dist_tensor.cpu().numpy()

    return predictions, decision_time, fit_time, unique_subspace_count, y_test

def __calculate_min_subspace_distances(subspaces, x_test_flattened):
    subspace_min_distance = []
    max_dist = x_test_flattened.shape[1] ** 0.5
    for point in x_test_flattened:
        min_distance = max_dist
        for subspace in subspaces:
            sub_dist = np.linalg.norm(point - subspace * point)
            min_distance = min(min_distance, sub_dist)
        subspace_min_distance.append(min_distance)
    return torch.Tensor(subspace_min_distance) / max_dist


def __calculate_occ_stats(y_test, dataset_type, predictions, decision_time, fit_time) -> dict:
    return {"Dataset": dataset_type,
            "AUC": auc(y_test, predictions),
            "PRAUC": average_precision_score(y_test, predictions),
            "F1": f1_score(y_test, (predictions > np.quantile(predictions, .80)) * 1),
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

def launch_outlier_detection_baseline(dataset_type: DatasetType, category: list[str], base_estimator, image_size=(224, 224), root_dir="../experiments/od_baselines", batch_size=512):

    x_train= load_data(dataset_type=dataset_type, category=category, image_size=image_size, train=True)

    x_train = extract_and_flatten_images_dataset_3d(x_train).cpu().numpy()
    x_train = normalize(x_train, axis=0)

    fit_time_start = time.time()
    base_estimator.fit(x_train)
    fit_time = time.time() - fit_time_start

    del x_train

    x_test, y_test = load_data(dataset_type=dataset_type, category=category, image_size=image_size, train=False)

    x_test = extract_and_flatten_images_dataset_3d(x_test).cpu().numpy()
    x_test = normalize(x_test, axis=0)

    n_samples = x_test.shape[0]
    decision_function_scores = np.zeros(n_samples)

    decision_time_start = time.time()

    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = x_test[i:end_idx]

        batch_scores = base_estimator.decision_function(batch)

        decision_function_scores[i:end_idx] = batch_scores

    decision_time = time.time() - decision_time_start

    del x_test

    stats = pd.DataFrame([__calculate_occ_stats(y_test, dataset_type, decision_function_scores, decision_time, fit_time)])
    print("Stats: ", stats)
    path_to_dir = Path(root_dir) / str(dataset_type.name)
    filename = base_estimator.__class__.__name__ + str(image_size[0]) + ".csv"
    os.makedirs(path_to_dir, exist_ok=True)
    stats.to_csv(path_to_dir / filename, index=False)
