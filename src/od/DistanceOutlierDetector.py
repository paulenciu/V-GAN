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
        self.train_score_max = None

    def fit(self, subspaces, x_train):
        self.subspaces = subspaces
        train_decision_scores = self.decision_score(x_train)
        self.train_score_max = np.max(train_decision_scores)

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

        self.decision_scores = torch.Tensor(subspace_min_distance) / max_dist
        self.decision_scores = self.scale_scores(self.decision_scores).cpu().numpy()
        self.decision_time = time.time() - decision_time_start
        return self.decision_scores

    def scale_scores(self, x):

        if self.train_score_max is None:
            return x

        return torch.nn.functional.sigmoid(x - self.train_score_max)

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
            raise NotImplementedError
        return dimensions ** 0.5
