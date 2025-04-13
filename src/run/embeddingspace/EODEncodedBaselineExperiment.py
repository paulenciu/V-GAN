import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyod.models.feature_bagging import FeatureBagging

from src.data.dataset.PreEmbeddedDataset import PreEmbeddedDataset
from src.data.dataset_loader import load_data

from sklearn.metrics import average_precision_score, f1_score
from sklearn.metrics import roc_auc_score as auc

class EODEncodedBaselineExperiment:

    def __init__(self, dataset_type, category, image_size_od, standardize_data, preprocessing_fn, od_models, encoder, encoder_name, root_dir="../experiments/od_baselines/embeddingspace/encoded"):
        self.dataset_type = dataset_type
        self.category = category
        self.image_size_od = image_size_od
        self.standardize_data = standardize_data
        self.preprocessing_fn = preprocessing_fn
        self.od_models = od_models
        self.root_dir = Path(root_dir) / encoder_name
        self.encoder = encoder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def fit(self):
        x_train, _ = load_data(dataset_type=self.dataset_type, category=self.category, image_size=self.image_size_od,
                            standardize=self.standardize_data)
        x_train = PreEmbeddedDataset(x_train, self.encoder, self.device)
        x_train_embeddings = x_train.embeddings
        x_train_embeddings = x_train_embeddings.view(x_train_embeddings.size(0), -1).detach().cpu().numpy()

        for od_model in self.od_models:
            od_model.fit(x_train_embeddings)

    def evaluate(self):
        x_test, y_test = load_data(dataset_type=self.dataset_type, category=self.category,
                                   image_size=self.image_size_od, standardize=self.standardize_data, train=False)
        x_test = PreEmbeddedDataset(x_test, self.encoder, self.device)
        x_test_embeddings = x_test.embeddings
        x_test_embeddings = x_test_embeddings.view(x_test_embeddings.size(0), -1).detach().cpu().numpy()

        y_test = np.array(y_test)

        for od_model in self.od_models:
            decision_scores = od_model.decision_function(x_test_embeddings)

            od_stats = pd.DataFrame([self.calculate_od_stats(od_model, y_test, decision_scores)])
            print("Stats: ", od_stats)
            path_to_dir = Path(self.root_dir) / str(self.dataset_type.name) / str(self.category)
            filename = od_model.__class__.__name__ + str(self.image_size_od[0]) + ".csv"
            os.makedirs(path_to_dir, exist_ok=True)

            file_path = path_to_dir / filename
            if file_path.exists():
                existing_data = pd.read_csv(file_path)
                combined_data = pd.concat([existing_data, od_stats], ignore_index=True)
                combined_data.to_csv(file_path, index=False)
            else:
                od_stats.to_csv(file_path, index=False)

    def calculate_od_stats(self, od_model, y_test, decision_scores):
        return {"Dataset": self.dataset_type,
                "AUC": auc(y_test, decision_scores),
                "PRAUC": average_precision_score(y_test, decision_scores),
                "F1": f1_score(y_test, (decision_scores > np.quantile(decision_scores, .80)) * 1),
                "OD Method": str(od_model.base_estimator.__class__.__name__) if isinstance(od_model, FeatureBagging) else str(od_model.__class__.__name__),}