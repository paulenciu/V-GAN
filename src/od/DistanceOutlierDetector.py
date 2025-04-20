import numpy as np
import torch
import time
from src.od.BaseOutlierDetection import BaseOutlierDetector
from src.utils.preprocessing import normalize_images_col_softmax, normalize_features, normalize_images


class DistanceOutlierDetector(BaseOutlierDetector):

    def __init__(self, preprocessing_fn):
        self.decision_time = 0.0
        self.subspaces = []
        self.decision_scores = None
        self.preprocessing_fn = preprocessing_fn

    def fit(self, subspaces, x_train):
        self.fit_time = 0
        self.subspaces = subspaces

    def decision_score(self, x_test):
        decision_time_start = time.time()
        subspace_min_distance = []
        max_dist = self.get_max_distance(x_test)
        for point in x_test:
            min_distance = max_dist
            for subspace in self.subspaces:
                sub_dist = np.linalg.norm(point - subspace * point)
                min_distance = min(min_distance, sub_dist)
            subspace_min_distance.append(min_distance)

        if max_dist == float("inf"):
            self.decision_scores = torch.Tensor(subspace_min_distance)
        else:
            self.decision_scores = torch.Tensor(subspace_min_distance) / max_dist
        self.decision_scores = self.decision_scores.cpu().numpy()
        self.decision_time = time.time() - decision_time_start
        return self.decision_scores

    def get_model_description(self):
        return {
            "Model": self.__class__.__name__,
        }

    def get_max_distance(self, x_test):
        if self.preprocessing_fn == normalize_features or self.preprocessing_fn == normalize_images_col_softmax:
            dimensions = x_test.shape[0]
        elif self.preprocessing_fn == normalize_images:
            dimensions = x_test.shape[1]
        else:
            return float("inf")
        return dimensions ** 0.5
