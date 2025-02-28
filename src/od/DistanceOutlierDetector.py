import numpy as np
import torch
import time
from src.od.BaseOutlierDetection import BaseOutlierDetector

class DistanceOutlierDetector(BaseOutlierDetector):

    def __init__(self):
        self.decision_time = 0.0
        self.subspaces = []
        self.decision_scores = None

    def fit(self, subspaces):
        self.subspaces = subspaces

    def decision_score(self, x_test):
        decision_time_start = time.time()
        subspace_min_distance = []
        max_dist = 10.000 ** 0.5
        for point in x_test:
            min_distance = max_dist
            for subspace in self.subspaces:
                sub_dist = np.linalg.norm(point - subspace * point)
                min_distance = min(min_distance, sub_dist)
            subspace_min_distance.append(min_distance)

        self.decision_scores = torch.Tensor(subspace_min_distance) / max_dist
        self.decision_scores.cpu().numpy()
        self.decision_time = time.time() - decision_time_start
        return self.decision_scores.cpu().numpy()

    def get_model_description(self):
        return {
            "Model": self.__class__.__name__,
        }
