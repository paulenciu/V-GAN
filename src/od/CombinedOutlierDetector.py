# src/od/combined_outlier_detector.py
import time
import numpy as np
from src.od.BaseOutlierDetection import BaseOutlierDetector


class CombinedOutlierDetector(BaseOutlierDetector):

    def __init__(self, weight_ensemble=0.5, base_estimators=None, max_n_jobs=4):
        self.weight_ensemble = weight_ensemble
        self.weight_distance = 1 - weight_ensemble

        self.ensemble_detector = EnsembleOutlierDetector(
            base_estimators=base_estimators,
            max_n_jobs=max_n_jobs
        )
        self.distance_detector = DistanceOutlierDetector()

        self.fit_time = 0.0
        self.decision_time = 0.0
        self.decision_scores = None

    def fit(self, subspaces, x_train):
        fit_start = time.time()

        self.ensemble_detector.fit(subspaces, x_train)
        self.distance_detector.fit(subspaces)

        self.fit_time = time.time() - fit_start
        return self

    def decision_function(self, x_test):
        decision_start = time.time()

        ensemble_scores = self.ensemble_detector.decision_score(x_test)
        distance_scores = self.distance_detector.decision_score(x_test)

        self.decision_scores = (self.weight_ensemble * ensemble_scores +
                                self.weight_distance * distance_scores)

        self.decision_time = time.time() - decision_start
        return self.decision_scores