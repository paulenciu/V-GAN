import time
import numpy as np
import random

from sel_suod.models.base import sel_SUOD
from src.data.dataset_loader import load_data

from torch.nn.functional import interpolate

from src.od.BaseOutlierDetection import BaseOutlierDetector

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

    def __init__(self, vmmd, base_estimators=None, max_n_jobs=4):
        self.base_estimators = base_estimators or []
        self.max_n_jobs = max_n_jobs
        self.ensemble_model = None
        self.train_time = 0.0
        self.decision_time = 0.0
        self.decision_scores_ens = None
        self.vmmd = vmmd
        self.n_subspaces = None

    def fit(self, subspaces, x):
        # Convert to float32 to save memory
        x = x.astype(np.float32)

        self.ensemble_model = sel_SUOD(base_estimators=self.base_estimators, subspaces=subspaces, n_jobs=self.max_n_jobs, bps_flag=False, approx_flag_global=False)
        fit_time_start = time.time()
        self.ensemble_model.fit(x)
        self.train_time = time.time() - fit_time_start
        self.n_subspaces = subspaces.shape[0]

    def decision_score(self, x, batch_size=512):
        n_samples = x.shape[0]
        decision_scores_ens = np.zeros(n_samples)

        batch_size = batch_size
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
        self.decision_scores_ens = decision_scores_ens
        return decision_scores_ens

    def get_model_description(self):
        return {
            "Model:": self.__class__.__name__,
            "Number Subspaces": str(self.n_subspaces),
            "Ensemble Model:": self.base_estimators[0].__class__.__name__,
        }
