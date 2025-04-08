import datetime

import numpy as np
import torch
from src.data.dataset.PreEmbeddedDataset import PreEmbeddedDataset
from src.data.dataset_loader import load_data
from src.od.CombinedOutlierDetector import CombinedOutlierDetector
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d, unflatten_images_3d
from src.utils.logger.od.EnsembleDetectionLogger import EnsembleDetectionLogger
from src.utils.logger.od.OutlierDetectionBenchmarkLogger import OutlierDetectionBenchmarkLogger
from src.utils.logger.od.OutlierDetectionLogger import OutlierDetectionLogger
from src.vmmd.VMMDEmbedding import VMMDEmbedding
from src.vmmd.VMMDWrapper import VMMDWrapper
from src.vmmd.outlier_detection.VMMDOD import VMMDOD
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score, f1_score

class EmbeddingOutlierDetectionExperiments:

    def __init__(self, vmmd: VMMDEmbedding, od_model: CombinedOutlierDetector, dataset_type, category, standardize_data=False, preprocessing_fn=lambda x: x, n_subspaces_sample=500):
        self.vmmd = vmmd
        self.od_model = od_model
        self.dataset_type = dataset_type
        self.category = category
        self.standardize_data = standardize_data
        self.preprocessing_fn = preprocessing_fn
        self.n_subspace_sample = n_subspaces_sample
        self.vmmd_od = VMMDOD(vmmd)
        self.vmmd_wrapper = VMMDWrapper(vmmd)
        self.od_logger = OutlierDetectionLogger(od_model, self.vmmd_od)
        self.od_bm_logger = OutlierDetectionBenchmarkLogger(od_model, self.vmmd_od, dataset_type=self.dataset_type,
                                                            category=self.category)
        self.ens_logger = EnsembleDetectionLogger(od_model.ensemble_detector, self.vmmd_od)
        self.image_size_train = (224, 224)

    def fit(self):
        x_train, _ = load_data(dataset_type=self.dataset_type, category=self.category, image_size=self.image_size_train,
                            standardize=self.standardize_data)
        self.vmmd.fit(dataset=x_train, preprocess_fn=self.preprocessing_fn)
        del x_train

        self.fit_outlier_detection()

    def fit_outlier_detection(self):
        self.vmmd.approx_subspace_dist(subspace_count=self.n_subspace_sample)
        subspaces = np.array(self.vmmd.subspaces, dtype=bool)

        print("Subspaces distribution:", self.vmmd.proba)

        # PREPARE DATA FOR OD
        x_train = load_data(dataset_type=self.dataset_type, category=self.category, image_size=self.image_size_train,
                            standardize=self.standardize_data)
        n_channels, height, width = x_train.image_shape
        x_train = extract_and_flatten_images_dataset_3d(x_train).cpu()
        x_train = torch.from_numpy(self.preprocessing_fn(x_train.numpy())).float()
        x_train = unflatten_images_3d(x_train, n_channels, height, width)
        x_train = PreEmbeddedDataset(x_train, self.vmmd.encoder, "cpu")
        self.od_model.fit(subspaces=subspaces, x_train=x_train.embeddings.detach().cpu().numpy())
        del x_train

    def evaluate_interval(self, ensemble_weight_start, ensemble_weight_end, step):
        # CALCULATE OD SCORES
        x_test, y_test = load_data(dataset_type=self.dataset_type, category=self.category,
                                   image_size=self.image_size_train, standardize=self.standardize_data, train=False)

        n_channels, height, width = x_test.image_shape
        x_test_flattened = extract_and_flatten_images_dataset_3d(x_test).to("cpu").numpy()
        x_test_flattened = torch.from_numpy(self.preprocessing_fn(x_test_flattened))
        x_test = unflatten_images_3d(x_test_flattened, n_channels, height, width)
        x_test = PreEmbeddedDataset(x_test, self.vmmd.encoder, "cpu")
        x_test_embeddings = x_test.embeddings.detach().cpu().numpy()
        y_test = np.array(y_test)

        self.ens_logger.log(x_test_embeddings, y_test)
        self.od_logger.log(x_test_embeddings, y_test)
        decision_scores, descriptions = self.od_model.decision_score_interval(x_test_embeddings, ensemble_weight_start,
                                                                              ensemble_weight_end, step)

        od_stats_list = []

        for i, ds in enumerate(decision_scores):
            od_stats = self.calculate_od_stats(y_test, ds)
            od_stats["OD Method"] = descriptions[i]
            self.vmmd_od.store_od_stats(od_stats, run_number=-1)
            od_stats_list.append(od_stats)

    def fit_pretrained_model(self, path_to_generator: str):
        self.vmmd_wrapper.load_model(path_to_generator)
        self.fit_outlier_detection()

    def calculate_od_stats(self, y_test, decision_scores):
        return {"Dataset": self.dataset_type,
                "AUC": auc(y_test, decision_scores),
                "PRAUC": average_precision_score(y_test, decision_scores),
                "F1": f1_score(y_test, (decision_scores > np.quantile(decision_scores, .80)) * 1),
                "Training Time": str(datetime.timedelta(seconds=self.od_model.fit_time)),
                "Decision Time": str(datetime.timedelta(seconds=self.od_model.decision_time)),
                "OD Method": self.od_model.get_model_description()}