import datetime
from pathlib import Path

import numpy as np
import os

import pandas as pd
from pyod.models.feature_bagging import FeatureBagging
from sel_suod.models.base import sel_SUOD
from sklearn.metrics import average_precision_score, f1_score
from sklearn.metrics import roc_auc_score as auc


from src.data.dataset_loader import load_data
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d


class OutlierDetectionBaselineExperiment:

    def __init__(self, dataset_type, category, image_size_od, standardize_data, preprocessing_fn, od_model, root_dir="../experiments/od_baselines"):
        self.dataset_type = dataset_type
        self.category = category
        self.image_size_od = image_size_od
        self.standardize_data = standardize_data
        self.preprocessing_fn = preprocessing_fn
        self.od_model = od_model
        self.root_dir = root_dir


    def fit(self):
        x_train = load_data(dataset_type=self.dataset_type, category=self.category, image_size=self.image_size_od,
                            standardize=self.standardize_data)
        x_train_flattened = extract_and_flatten_images_dataset_3d(x_train).to("cpu").numpy()
        x_train_flattened = self.preprocessing_fn(x_train_flattened)
        self.od_model.fit(x_train_flattened)
        del x_train

    def evaluate(self):
        x_test, y_test = load_data(dataset_type=self.dataset_type, category=self.category,
                                   image_size=self.image_size_od, standardize=self.standardize_data, train=False)
        x_test_flattened = extract_and_flatten_images_dataset_3d(x_test).to("cpu").numpy()
        x_test_flattened = self.preprocessing_fn(x_test_flattened)
        y_test = np.array(y_test)
        decision_scores = self.od_model.decision_function(x_test_flattened)

        od_stats = pd.DataFrame([self.calculate_od_stats(y_test, decision_scores)])
        print("Stats: ", od_stats)
        path_to_dir = Path(self.root_dir) / str(self.dataset_type.name) / str(self.category)
        filename = self.od_model.__class__.__name__ + str(self.image_size_od[0]) + ".csv"
        os.makedirs(path_to_dir, exist_ok=True)
        od_stats.to_csv(path_to_dir / filename, index=False)

    def calculate_od_stats(self, y_test, decision_scores):
        return {"Dataset": self.dataset_type,
                "AUC": auc(y_test, decision_scores),
                "PRAUC": average_precision_score(y_test, decision_scores),
                "F1": f1_score(y_test, (decision_scores > np.quantile(decision_scores, .80)) * 1),
                "OD Method": self.od_model.__class__.__name__ + str(self.od_model.n_estimators) if isinstance(self.od_model, FeatureBagging) else "" + self.od_model.base_estimator.__class__.__name__ if isinstance(self.od_model, FeatureBagging) else "",}