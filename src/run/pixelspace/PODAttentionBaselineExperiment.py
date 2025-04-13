import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyod.models.feature_bagging import FeatureBagging

from src.data.dataset_loader import load_data
from src.models.encoder.IdentityEncoder import IdentityEncoder
from src.utils.BigUBuilder import calculate_average_u
from src.vmmd.VMMD import VMMD
from src.vmmd.VMMDWrapper import VMMDWrapper
from sklearn.metrics import average_precision_score, f1_score
from sklearn.metrics import roc_auc_score as auc

from src.vmmd.model.VMMDDiagonal1Channel import VMMDDiagonal1Channel


class PODAttentionBaselineExperiment:

    def __init__(self, dataset_type, category, image_size_od, standardize_data, preprocessing_fn, od_models, root_dir="../experiments/od_baselines/pixelspace/attention", exp_date="21-03"):
        self.dataset_type = dataset_type
        self.category = category
        self.image_size_od = image_size_od
        self.standardize_data = standardize_data
        self.preprocessing_fn = preprocessing_fn
        self.od_models = od_models
        self.root_dir = root_dir
        self.exp_date = exp_date
        self.vmmd = VMMDDiagonal1Channel(filename="placeholder_name", autoencoder=IdentityEncoder(), generator=None)
        self.vmmd_wrapper = VMMDWrapper(self.vmmd)

    def fit(self):
        x_train, _ = load_data(dataset_type=self.dataset_type, category=self.category, image_size=self.image_size_od,
                            standardize=self.standardize_data)

        attention_mask = self.create_attention_mask()
        attention_mask = attention_mask.unsqueeze(0).repeat(x_train.shape[0], 1, 1, 1)
        x_train = self.vmmd.apply_subspaces_operator(x_train, attention_mask)

        x_train = x_train.view(x_train.shape[0], -1).cpu().numpy()
        x_train = self.preprocessing_fn(x_train)

        for od_model in self.od_models:
            od_model.fit(x_train)
        del x_train

    def evaluate(self):
        x_test, y_test = load_data(dataset_type=self.dataset_type, category=self.category,
                                   image_size=self.image_size_od, standardize=self.standardize_data, train=False)

        attention_mask = self.create_attention_mask()
        attention_mask = attention_mask.unsqueeze(0).repeat(x_test.shape[0], 1, 1, 1)
        x_test = self.vmmd.apply_subspaces_operator(x_test, attention_mask)

        x_test = x_test.view(x_test.shape[0], -1).cpu().numpy()
        x_test = self.preprocessing_fn(x_test)
        y_test = np.array(y_test)

        for od_model in self.od_models:
            decision_scores = od_model.decision_function(x_test)

            od_stats = pd.DataFrame([self.calculate_od_stats(od_model, y_test, decision_scores)])
            print("Stats: ", od_stats)
            path_to_dir = Path(self.root_dir) / str(self.dataset_type.name) / str(self.category)
            filename = od_model.__class__.__name__ + str(self.image_size_od[0]) + ".csv"

            file_path = path_to_dir / filename
            if file_path.exists():
                existing_data = pd.read_csv(file_path)
                combined_data = pd.concat([existing_data, od_stats], ignore_index=True)
                combined_data.to_csv(file_path, index=False)
            else:
                os.makedirs(str(path_to_dir), exist_ok=True)
                od_stats.to_csv(file_path, index=False)

    def calculate_od_stats(self, od_model, y_test, decision_scores):
        return {"Dataset": self.dataset_type,
                "AUC": auc(y_test, decision_scores),
                "PRAUC": average_precision_score(y_test, decision_scores),
                "F1": f1_score(y_test, (decision_scores > np.quantile(decision_scores, .80)) * 1),
                "OD Method": str(od_model.base_estimator.__class__.__name__) if isinstance(od_model, FeatureBagging) else str(od_model.__class__.__name__),}

    def create_attention_mask(self):
        root_dir = Path("../experiments/remote/") / self.exp_date
        model_param_path = None
        for dir_name in os.listdir(root_dir):
            if dir_name.startswith(self.dataset_type.name) and dir_name.__contains__(self.category):
                model_path =  root_dir / dir_name / "models"
                fname =  f"generator_{len(os.listdir(model_path)) - 1}.pt"
                model_param_path = model_path / fname
        if model_param_path is None:
            raise FileNotFoundError(f"No model found in {root_dir}")

        self.vmmd_wrapper.load_model(str(model_param_path))

        u = self.vmmd.sample_count_subspaces(count=500)
        u_avg = calculate_average_u(u)
        return u_avg




