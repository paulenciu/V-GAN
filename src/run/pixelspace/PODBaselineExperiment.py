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


class PODBaselineExperiment:

    def __init__(self, dataset_type, category, image_size_od, standardize_data, preprocessing_fn, od_model, root_dir="../experiments/od_baselines/pixelspace"):
        self.dataset_type = dataset_type
        self.category = category
        self.image_size_od = image_size_od
        self.standardize_data = standardize_data
        self.preprocessing_fn = preprocessing_fn
        self.od_model = od_model
        self.root_dir = root_dir


    def fit(self):
        x_train, _ = load_data(dataset_type=self.dataset_type, category=self.category, image_size=self.image_size_od,
                            standardize=self.standardize_data)
        x_train = x_train.view(x_train.shape[0], -1).cpu().numpy()
        x_train = self.preprocessing_fn(x_train)
        self.od_model.fit(x_train)
        del x_train

    def evaluate(self):
        x_test, y_test = load_data(dataset_type=self.dataset_type, category=self.category,
                                   image_size=self.image_size_od, standardize=self.standardize_data, train=False)
        x_test = x_test.view(x_test.shape[0], -1).cpu().numpy()
        x_test = self.preprocessing_fn(x_test)
        y_test = np.array(y_test)
        decision_scores = self.od_model.decision_function(x_test)

        od_stats = pd.DataFrame([self.calculate_od_stats(y_test, decision_scores)])
        print("Stats: ", od_stats)
        path_to_dir = Path(self.root_dir) / str(self.dataset_type.name) / str(self.category)
        filename = self.od_model.__class__.__name__ + str(self.image_size_od[0]) + ".csv"

        file_path = path_to_dir / filename
        if file_path.exists():
            existing_data = pd.read_csv(file_path)
            combined_data = pd.concat([existing_data, od_stats], ignore_index=True)
            combined_data.to_csv(file_path, index=False)
        else:
            od_stats.to_csv(file_path, index=False)

    def calculate_od_stats(self,y_test, decision_scores):
        return {"Dataset": self.dataset_type,
                "AUC": auc(y_test, decision_scores),
                "PRAUC": average_precision_score(y_test, decision_scores),
                "F1": f1_score(y_test, (decision_scores > np.quantile(decision_scores, .80)) * 1),
                "OD Method": str(self.od_model.base_estimator.__class__.__name__) if isinstance(self.od_model, FeatureBagging) else str(self.od_model.__class__.__name__),}