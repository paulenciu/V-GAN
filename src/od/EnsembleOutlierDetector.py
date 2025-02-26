import time
import numpy as np

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

    def __init__(self, base_estimators=None, max_n_jobs=4):
        self.base_estimators = base_estimators or []
        self.max_n_jobs = max_n_jobs
        self.ensemble_model = None
        self.train_time = 0.0
        self.decision_time = 0.0
        self.decision_scores_ens = None

    def fit(self, subspaces, x_train):
        self.ensemble_model = sel_SUOD(base_estimators=self.base_estimators, subspaces=subspaces, n_jobs=self.max_n_jobs, bps_flag=False, approx_flag_global=False)
        x_train = load_data(dataset_type=self.dataset_type, category=self.category, image_size=self.image_size_test, standardize=self.standardize_data)

        fit_time_start = time.time()
        self.ensemble_model.fit(x_train_flattened)
        self.train_time = time.time() - fit_time_start


    def decision_score(self, x_test, y_test, batch_size=512):
        n_samples = x_test.shape[0]
        decision_scores_ens = np.zeros(n_samples)

        batch_size = batch_size
        decision_time_start = time.time()

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = x_test[i:end_idx]
            batch_scores = self.ensemble_model.decision_function(batch)
            decision_scores_ens[i:end_idx] = batch_scores

        decision_scores_ens = aggregator_funct(
                decision_scores_ens,
                weights=self.vmmd.proba,
                type="avg"
            )

        self.decision_time = time.time() - decision_time_start
        self.decision_scores_ens = decision_scores_ens
