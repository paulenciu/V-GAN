import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from src.data.dataset_loader import load_data
from src.od.CombinedOutlierDetector import CombinedOutlierDetector
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d
from src.utils.Plotter import tensor_to_image
from src.utils.logger.od.EnsembleDetectionLogger import EnsembleDetectionLogger
from src.utils.logger.od.OutlierDetectionBenchmarkLogger import OutlierDetectionBenchmarkLogger
from src.utils.preprocessing import no_preprocessing
from src.vmmd.VMMDWrapper import VMMDWrapper
from src.utils.logger.od.OutlierDetectionLogger import OutlierDetectionLogger
from src.vmmd.outlier_detection.VMMDOD import VMMDOD
from torch.nn.functional import interpolate
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score, f1_score

class OutlierDetectionExperiment:

    def __init__(self, vmmd, od_model: CombinedOutlierDetector, dataset_type, category, image_size_train, standardize_data=False, image_size_od=None, preprocessing_fn=no_preprocessing, n_subspaces_sample=500):
        self.vmmd = vmmd
        self.vmmd_od = VMMDOD(vmmd)
        self.vmmd_wrapper = VMMDWrapper(vmmd)
        self.dataset_type = dataset_type
        self.category = category
        self.image_size_train = image_size_train
        self.image_size_od = image_size_od or image_size_train
        self.standardize_data = standardize_data
        self.od_model = od_model
        self.od_logger = OutlierDetectionLogger(od_model, self.vmmd_od)
        self.od_bm_logger = OutlierDetectionBenchmarkLogger(od_model, self.vmmd_od, dataset_type=self.dataset_type, category=self.category)
        self.ens_logger = EnsembleDetectionLogger(od_model.ensemble_detector, self.vmmd_od)
        self.preprocessing_fn = preprocessing_fn
        self.n_subspace_sample = n_subspaces_sample

    def fit(self):
        x_train, _ = load_data(dataset_type=self.dataset_type, category=self.category, image_size=self.image_size_train, standardize=self.standardize_data)
        self.vmmd.fit(dataset=x_train, preprocess_fn=self.preprocessing_fn)
        del x_train

        self.fit_outlier_detection()

    def fit_outlier_detection(self):
        self.vmmd.approx_subspace_dist(subspace_count=self.n_subspace_sample)
        subspaces = self.vmmd.subspaces

        print("Subspaces distribution:", self.vmmd.proba)

        # PREPARE SUBSPACE FOR OD
        subspaces = interpolate(subspaces, size=self.image_size_od[0], mode='nearest')
        subspaces = subspaces.view(subspaces.shape[0], -1)

        print("Number of unique subspaces:", len(subspaces), "/", self.n_subspace_sample)
        subspaces = np.array(subspaces, dtype=bool)
        self.vmmd_od.store_subspaces(subspaces)

        # PREPARE DATA FOR OD
        x_train, _ = load_data(dataset_type=self.dataset_type, category=self.category, image_size=self.image_size_od,
                            standardize=self.standardize_data)
        x_train_flattened = x_train.view(x_train.shape[0], -1).cpu().numpy()
        x_train_flattened = self.preprocessing_fn(x_train_flattened)

        self.od_model.fit(subspaces, x_train_flattened)
        del x_train_flattened

    def fit_pretrained_model(self, path_to_generator: str):
        self.vmmd_wrapper.load_model(path_to_generator)
        self.fit_outlier_detection()

    def evaluate(self, store_stats=True, weight_ensemble=0.5):
        # CALCULATE OD SCORES
        x_test, y_test = load_data(dataset_type=self.dataset_type, category=self.category,
                                   image_size=self.image_size_od, standardize=self.standardize_data, train=False)
        x_test_flattened = x_test.view(x_test.shape[0], -1).cpu().numpy()
        x_test_flattened = self.preprocessing_fn(x_test_flattened)
        y_test = np.array(y_test)

        self.od_model.update_tradeoff(weight_ensemble=weight_ensemble)
        decision_scores = self.od_model.decision_score(x_test_flattened)

        od_stats = self.calculate_od_stats(y_test, decision_scores)
        return od_stats if not store_stats else self.vmmd_od.store_od_stats(od_stats, run_number=-1)

    def evaluate_interval(self, ensemble_weight_start, ensemble_weight_end, step):
        # CALCULATE OD SCORES
        x_test, y_test = load_data(dataset_type=self.dataset_type, category=self.category,
                                   image_size=self.image_size_od, standardize=self.standardize_data, train=False)
        x_test_flattened = x_test.view(x_test.shape[0], -1).cpu().numpy()
        x_test_flattened = self.preprocessing_fn(x_test_flattened)
        y_test = np.array(y_test)

        #self.ens_logger.log(x_test_flattened, y_test)
        self.od_logger.log(x_test_flattened, y_test)
        decision_scores, descriptions = self.od_model.decision_score_interval(x_test_flattened, ensemble_weight_start, ensemble_weight_end, step)

        od_stats_list = []

        for i, ds in enumerate(decision_scores):
            od_stats = self.calculate_od_stats(y_test, ds)
            od_stats["OD Method"] = descriptions[i]
            self.vmmd_od.store_od_stats(od_stats, run_number=-1)
            od_stats_list.append(od_stats)
        interval_length = (ensemble_weight_end - ensemble_weight_start)  * (1 / step) + 1
#        self.od_bm_logger.log(od_stats_list, interval_length=interval_length)

    def calculate_od_stats(self, y_test, decision_scores):
        return {"Dataset": self.dataset_type,
                "AUC": auc(y_test, decision_scores),
                "PRAUC": average_precision_score(y_test, decision_scores),
                "F1": f1_score(y_test, (decision_scores > np.quantile(decision_scores, .80)) * 1),
                "Training Time": str(datetime.timedelta(seconds=self.od_model.fit_time)),
                "Decision Time": str(datetime.timedelta(seconds=self.od_model.decision_time)),
                "OD Method": self.od_model.get_model_description()}