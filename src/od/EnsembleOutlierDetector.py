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
        self.ensemble_model = None
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

    def fit(self, subspaces, x):
        x = x.astype(np.float32)

        self.ensemble_model = sel_SUOD(base_estimators=self.base_estimators, subspaces=subspaces, n_jobs=self.max_n_jobs, bps_flag=False, approx_flag_global=False)
        fit_time_start = time.time()
        self.ensemble_model.fit(x)
        self.train_time = time.time() - fit_time_start

        #NORMALIZE SCORES USING TRAINING DATA STATISTICS
        train_scores = self.ensemble_model.decision_function(x)
        train_scores_agg = aggregator_funct(
            train_scores, weights=self.vmmd.proba, type="avg"
        )

        self.ens_train_min = np.min(train_scores_agg)
        self.ens_train_max = np.percentile(train_scores_agg, 95)
        self.ens_train_score_std = np.std([x for x in train_scores_agg if x <= self.ens_train_max]) + 1e-10
        print("Ensemble train score max: ", self.ens_train_max, "Ensemble train score std: ", self.ens_train_score_std)
        self.n_subspaces = subspaces.shape[0]

    def decision_score(self, x, batch_size=512):
        n_samples = x.shape[0]
        decision_scores_ens = np.zeros(n_samples)

        decision_time_start = time.time()
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = x[i:end_idx]
            batch_scores = self.ensemble_model.decision_function(batch)

            decision_scores_ens[i:end_idx] = aggregator_funct(
                    batch_scores,
                    weights=self.vmmd.proba,
                    type="avg"
            )

        self.decision_time = time.time() - decision_time_start
        self.decision_scores_ens = self.scale_scores(decision_scores_ens)
        return self.decision_scores_ens

    def scale_scores(self, x):
        #return 1 / (1 + np.exp(-(x - self.ens_train_max) / self.ens_train_score_std))
        return x

    def get_model_description(self):
        return {
            "Model:": self.__class__.__name__,
            "Number Subspaces": str(self.n_subspaces),
            "Ensemble Model:": self.base_estimators[0].__class__.__name__,
        }
