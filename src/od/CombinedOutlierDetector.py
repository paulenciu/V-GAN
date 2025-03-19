import time
import numpy as np
from src.od.BaseOutlierDetection import BaseOutlierDetector
from src.od.DistanceOutlierDetector import DistanceOutlierDetector
from src.od.EnsembleOutlierDetector import EnsembleOutlierDetector
from src.utils.preprocessing import min_max_scaling


class CombinedOutlierDetector(BaseOutlierDetector):

    def __init__(self, vmmd, preprocessing_fn, weight_ensemble=0.5, base_estimators=None, max_n_jobs=4):
        super().__init__()
        self.weight_ensemble = weight_ensemble
        self.weight_distance = 1 - weight_ensemble

        self.ensemble_detector = EnsembleOutlierDetector(
            base_estimators=base_estimators,
            max_n_jobs=max_n_jobs,
            vmmd=vmmd,
        )
        self.distance_detector = DistanceOutlierDetector(
            preprocessing_fn=preprocessing_fn
        )

        self.fit_time = 0.0
        self.decision_time = 0.0
        self.decision_scores = None

    def update_tradeoff(self, weight_ensemble):
        self.weight_ensemble = weight_ensemble
        self.weight_distance = 1 - weight_ensemble

    def get_ensemble_score(self, x):
        return self.ensemble_detector.decision_score_agg(x)

    def get_distance_score(self, x):
        return self.distance_detector.decision_score(x)

    def fit(self, subspaces, x_train):
        fit_start = time.time()

        print("Ensemble Outlier Detector Fit")
        self.ensemble_detector.fit(subspaces, x_train)

        print("Distance Outlier Detector Fit")
        self.distance_detector.fit(subspaces, x_train)

        self.fit_time = time.time() - fit_start
        return self

    def decision_score_interval(self, x_test, ensemble_weight_start, ensemble_weight_end, step):
        ensemble_scores = self.ensemble_detector.decision_score_agg(x_test)
        distance_scores = self.distance_detector.decision_score(x_test)

        decision_scores_list = []
        description_list = []

        for ensemble_weight in np.arange(ensemble_weight_start, ensemble_weight_end + step, step):
            self.update_tradeoff(ensemble_weight)
            decision_scores = self.weight_ensemble * ensemble_scores + self.weight_distance * distance_scores
            description = self.get_model_description()

            decision_scores_list.append(decision_scores)
            description_list.append(description)

        return decision_scores_list, description_list

    def decision_score(self, x_test):
        decision_start = time.time()

        if self.weight_ensemble == 0:
            distance_scores = self.distance_detector.decision_score(x_test)
            distance_scores = min_max_scaling(distance_scores)
            self.decision_time = time.time() - decision_start
            return distance_scores

        elif self.weight_ensemble == 1:
            ensemble_scores = self.ensemble_detector.decision_score_agg(x_test)
            ensemble_scores = min_max_scaling(ensemble_scores)
            self.decision_time = time.time() - decision_start
            return ensemble_scores

        ensemble_scores = self.ensemble_detector.decision_score_agg(x_test)
        distance_scores = self.distance_detector.decision_score(x_test)

        ensemble_scores = min_max_scaling(ensemble_scores)
        distance_scores = min_max_scaling(distance_scores)

        self.decision_scores = self.weight_ensemble * ensemble_scores + self.weight_distance * distance_scores

        self.decision_time = time.time() - decision_start
        return self.decision_scores

    def get_model_description(self):
        ensemble_description = self.ensemble_detector.get_model_description()
        distance_description = self.distance_detector.get_model_description()

        ensemble_description["ensemble weight"] = self.weight_ensemble
        distance_description["distance weight"] = self.weight_distance

        return {
            "Ensemble Description": ensemble_description,
            "Distance Description": distance_description
        }