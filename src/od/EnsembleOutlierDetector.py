import time
import numpy as np
import random

import torch.nn.functional
from sel_suod.models.base import sel_SUOD
from sklearn.preprocessing import MinMaxScaler

from src.data.dataset_loader import load_data

from torch.nn.functional import interpolate

from src.od.BaseOutlierDetection import BaseOutlierDetector
from src.utils.preprocessing import min_max_scaling


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

class EnsembleOutlierDetector(BaseOutlierDetector):

    def __init__(self, vmmd, base_estimators=None, max_n_jobs=4, k = 1, temperature = 1):
        self.base_estimators = base_estimators or []
        self.max_n_jobs = max_n_jobs
        self.ensemble_models = []
        self.train_time = 0.0
        self.decision_time = 0.0
        self.decision_scores_ens = None
        self.vmmd = vmmd
        self.n_subspaces = None
        self.ens_train_min = None
        self.ens_train_max = None
        self.ens_train_score_std = None
        self.k = k
        self.temperature = temperature
        self.train_times = []
        self.decision_times = []
        self.model_description = []

    def fit(self, subspaces, x):
        x = x.astype(np.float32)
        for base_estimator in self.base_estimators:
            self.ensemble_models.append(
                sel_SUOD(
                    base_estimators=[base_estimator],
                    subspaces=subspaces,
                    n_jobs=self.max_n_jobs,
                    bps_flag=False,
                    approx_flag_global=False
                )
            )

        for ensemble_model in self.ensemble_models:
            fit_time_start = time.time()
            ensemble_model.fit(x)
            self.train_times.append(time.time() - fit_time_start)

        self.n_subspaces = subspaces.shape[0]

    def decision_score(self, x):
        decision_scores = []
        for ensemble_model in self.ensemble_models:
            decision_time_start = time.time()
            decision_scores.append(ensemble_model.decision_function(x))
            self.decision_times.append(time.time() - decision_time_start)
        return decision_scores

    def decision_score_agg(self, x):
        decision_score_ens = []
        for decision_score in self.decision_score(x):
            decision_score_ens.append(
                aggregator_funct(
                    decision_score,
                    weights=self.vmmd.proba,
                    type="avg"
                )
            )
        self.decision_scores_ens = [self.scale_scores(dc_ens) for dc_ens in decision_score_ens]
        return self.decision_scores_ens

    def scale_scores(self, x):
        return x

    def get_model_description(self, idx=0):
        return {
            "Model": self.__class__.__name__,
            "Number Subspaces": str(self.n_subspaces),
            "Ensemble Model": self.base_estimators[idx].__class__.__name__,
        }
