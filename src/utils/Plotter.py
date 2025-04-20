import ast
import os

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset_type import DatasetType
from src.run.pipeline.BaselinePipeline import mvtec_categories, cifar10_classes, fashionmnist_categories


def extract_name_and_score_from_row(row, metric):
    od_model = row["OD Method"]
    od_dict = ast.literal_eval(od_model)
    return od_dict["Ensemble Description"]["Ensemble Model"], row[metric.upper()]

def is_fs_model(row):
    od_model = row["OD Method"]
    od_dict = ast.literal_eval(od_model)
    return od_dict["Ensemble Description"]["ensemble weight"] == 1.0

def plot_ens_dis_comparison(exp_date, metric="auc", model=""):

    config = [
        (DatasetType.MVTEC_AD, mvtec_categories),
        (DatasetType.OCCCIFAR10, cifar10_classes),
        (DatasetType.OCCFMNIST, fashionmnist_categories)
    ]

    root_dir = Path("../experiments/remote/") / exp_date

    x = np.arange(0, 1.1, 0.1)

    dataset_results = {}

    for dataset_type, categories in config:
        y_scores = {}

        for category in categories:
            prefix = dataset_type.name + "[" + category.split("/")[0]

            for fname in os.listdir(root_dir):

                if model == "":
                    if fname.startswith(prefix) and fname.endswith(model):
                        score_df = pd.read_csv(root_dir / fname / "od_stats_-1.csv")

                        method_scores = []
                        scores_total = {}
                        for _, row in score_df.iterrows():

                            name, score = extract_name_and_score_from_row(row, metric)
                            method_scores.append(score)

                            if is_fs_model(row):
                                scores_total[name] = method_scores
                                method_scores = []

                        y_scores[category] = scores_total
                        break
                else:
                    if fname.__contains__(prefix) and fname.endswith(model):
                        score_df = pd.read_csv(root_dir / fname / "od_stats_-1.csv")

                        method_scores = []
                        scores_total = {}
                        for _, row in score_df.iterrows():

                            name, score = extract_name_and_score_from_row(row, metric)
                            method_scores.append(score)

                            if is_fs_model(row):
                                scores_total[name] = method_scores
                                method_scores = []

                        y_scores[category] = scores_total
                        break
        dataset_results[dataset_type.name] = y_scores

    for dataset_name, y_scores in dataset_results.items():
        plt.figure(figsize=(10, 6))
        all_methods = []
        for method in ["LUNAR", "LOF", "KNN"]:
            all_scores = []
            for category in y_scores.keys():
                if method in y_scores[category]:
                    scores = y_scores[category][method]
                    if len(scores) == len(x):  # Ensure same length as x
                        all_scores.append(scores)

            if all_scores:
                mean_scores = np.mean(all_scores, axis=0)
                plt.plot(x, mean_scores, marker='o', label=method)
                all_methods.append(mean_scores)

        if all_methods:
            mean_scores = np.mean(all_methods, axis=0)
            plt.plot(x, mean_scores, marker='o', label="mean")

        plt.title(f'R50 {dataset_name} - {metric.upper()} Performance Comparison')
        plt.xlabel('Threshold')
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()