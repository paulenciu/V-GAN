from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.data.dataset_type import DatasetType
from src.od import CombinedOutlierDetector
from src.vmmd.outlier_detection import VMMDOD


class OutlierDetectionBenchmarkLogger:

    def __init__(self, od_model: CombinedOutlierDetector, vmmd_od: VMMDOD, dataset_type: DatasetType, category: str):
        self.od_model = od_model
        self.vmmd_od = vmmd_od
        self.dataset_type = dataset_type
        self.category = category

    def log(self, scores: list[dict], interval_length=11) -> None:

        ens_model_idxs = np.arange(interval_length - 1, len(scores), interval_length, dtype=int)
        dis_model_idxs = np.arange(0, len(scores), interval_length - 1, dtype=int)
        ens_model_scores = [scores[i] for i in ens_model_idxs.tolist()]
        dis_scores = scores[0]
        baseline_scores = self.get_baseline_score()

        comb_score_idx = np.setdiff1d(np.arange(len(scores)), np.concatenate([ens_model_idxs, dis_model_idxs]))
        comb_scores = [scores[i] for i in comb_score_idx.tolist()]
        best_comb_scores = self.find_best_combinational_score(comb_scores)

        all_scores = [dis_scores, best_comb_scores] + ens_model_scores  + baseline_scores
        model_names = ["VGAN + ERROR"] + ["VGAN + " +  best_comb_scores["OD Method"]["Ensemble Description"]["Ensemble Model"]  + " + ERROR"] +  ["VGAN + " + ens_scores["OD Method"]["Ensemble Description"]["Ensemble Model"] for ens_scores in ens_model_scores] + [score["OD Method"] for score in baseline_scores]
        metrics = ["AUC", "PRAUC", "F1"]

        n_models = len(all_scores)
        n_metrics = len(metrics)
        x = np.arange(n_metrics)

        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.1

        for i, model_name in enumerate(model_names):
            metric_values = [all_scores[i][metric] for metric in metrics]
            bars = ax.bar(x + i * bar_width, metric_values, bar_width, label=model_name)

            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        for i, model_name in enumerate(model_names):
            for j, metric in enumerate(metrics):
                ax.text(
                    x[j] + i * bar_width,
                    -0.05,
                    model_name,
                    ha="center",
                    va="top",
                    fontsize=9,
                    rotation=90,
                )

        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + bar_width * (n_models - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend().remove()

        plt.tight_layout()
        plt.show()
        self.vmmd_od.store_od_benchmarks(fig)

    def get_baseline_score(self):
        baseline_scores = []
        path_to_baseline = Path("../experiments/od_baselines/") / str(self.dataset_type.name) / str(self.category)
        for file in path_to_baseline.iterdir():
            if file.name.endswith(".csv"):
                baseline_score = {}
                row = pd.read_csv(file)
                baseline_score["OD Method"] = file.stem
                baseline_score["AUC"] = row['AUC'].item()
                baseline_score["PRAUC"] = row['PRAUC'].item()
                baseline_score["F1"] = row['F1'].item()
                baseline_scores.append(baseline_score)

        if len(baseline_scores) == 0:
            self.initiate_baseline_experiment()
            return self.get_baseline_score()
        return baseline_scores

    def initiate_baseline_experiment(self):
        raise NotImplementedError

    def find_best_combinational_score(self, comb_scores: list[dict]):
        return max(comb_scores, key=lambda entry: (entry["AUC"], entry["PRAUC"], entry["F1"]))