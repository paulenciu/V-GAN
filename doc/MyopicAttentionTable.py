import os
import ast
from collections import defaultdict
from itertools import count
from pathlib import Path

import pandas as pd

from doc.AbstractTable import AbstractTable
from src.data import dataset_type
from src.data.dataset_type import DatasetType
from src.run.pipeline.BaselinePipeline import mvtec_categories, cifar10_classes, fashionmnist_categories


def print_latex_table(latex_code, metric="auc"):
    modified_code = (
        "\\begin{table*}\n"
        "\\centering\n"
        "\\caption{Myopic Attention " + metric.upper() + "}\n"
        "\\label{myopic_attention" + metric.lower() + "}\n"
        + latex_code +
        "\\end{table*}"
    )

    modified_code = modified_code.replace(
        r'\begin{tabular}{llllllllllllllll}',
        r'\begin{tabular}{l|ccc|ccc|ccc|cc|cc|cc}'
    )

    # Center method headers
    for method in ["LUNAR", "LOF", "KNN"]:
        modified_code = modified_code.replace(
            rf'\multicolumn{{3}}{{r}}{{{method}}}',
            rf'\multicolumn{{3}}{{c}}{{{method}}}'
        )

    for method in ["PADIM", "STFPM", "DFM"]:
        modified_code = modified_code.replace(
            rf'\multicolumn{{2}}{{r}}{{{method}}}',
            rf'\multicolumn{{2}}{{c}}{{{method}}}'
        )

    print(modified_code)



class MyopicAttentionTable(AbstractTable):

    DATASET_CONFIG = [
        (DatasetType.MVTEC_AD, "MVTec AD", mvtec_categories),
        (DatasetType.OCCCIFAR10, "Cifar10", cifar10_classes),
        (DatasetType.OCCFMNIST, "FashionMNIST", fashionmnist_categories)
    ]

    METHOD_CONFIG = [
        ("LUNAR", ["FuS", "FeB", "FuS+A"]),
        ("LOF", ["FuS", "FeB", "FuS+A"]),
        ("KNN", ["FuS", "FeB", "FuS+A"]),
        ("PADIM", ["FuS", "FuS+A"]),
        ("DFM", ["FuS", "FuS+A"]),
        ("STFPM", ["FuS", "FuS+A"])
    ]

    def get_fs_baseline_scores(self, dataset_type, category, metric, att=False):
        # GETTING FS VALUES OF LUNAR, LOF, and KNN
        fs_methods = ["LUNAR", "LOF", "KNN"]

        if att:
            root_dir = Path(
                "../experiments/od_baselines/pixelspace/attention/normalized") / dataset_type.name / category
            anomalib_benchmark_file = Path(
                "../experiments/od_baselines/pixelspace/attention/normalized") / dataset_type.name / "anomaly_benchmarks_attention.csv"

        else:
            root_dir = Path(
                "../experiments/od_baselines/pixelspace") / dataset_type.name / category
            anomalib_benchmark_file = Path(
                "../experiments/od_baselines/pixelspace") / dataset_type.name / "anomaly_benchmarks.csv"

        scores = {}
        for fname in os.listdir(root_dir):
            if any([fname.startswith(prefix) for prefix in fs_methods]):
                method_name = self.extract_method_name(to_extract_from=fs_methods, to_search_in=fname)
                score_df = pd.read_csv(root_dir / fname)
                score = score_df[metric.upper()].values[0]
                scores[method_name] = (f"{score:.3f}")

        # GETTING anomalib scores
        fs_methods = ["PADIM", "DFM", "STFPM"]
        if metric == "auc" and dataset_type:
            anomalib_df = pd.read_csv(anomalib_benchmark_file)
            category_rows = anomalib_df[
                anomalib_df["category"].str.lower() == category.lower()
                ]
            for method_name in fs_methods:
                model_row = category_rows[category_rows["model"].str.lower() == method_name.lower()]
                score = model_row["auroc"].values[0]
                scores[method_name] = (f"{score:.3f}")
        else:
            for method_name in fs_methods:
                scores[method_name] = "NA"

        return scores

    def get_ens_baseline_scores(self, dataset_type, category, metric):
        root_dir = Path("../experiments/od_baselines/pixelspace") / dataset_type.name / category
        methods = ["LUNAR", "LOF", "KNN"]
        scores = {}
        for fname in os.listdir(root_dir):
            if fname.startswith("FeatureBagging"):
                score_df = pd.read_csv(root_dir / fname)

                for method_name in methods:
                    row = score_df[score_df["OD Method"].str.lower() == method_name.lower()]
                    scores[method_name]= f"{row[metric.upper()].values[0]:.3f}"

                return scores

        return {k: "NA" for k in methods}

    def generate_table(self, metric="auc"):

        cols = [("", "")]

        for method, spaces in self.METHOD_CONFIG:
            for space in spaces:
                cols.append((method, space))


        columns = pd.MultiIndex.from_tuples(cols)
        data = []
        for dataset_type, dataset_name, categories in self.DATASET_CONFIG:
            data.extend([["\\textbf{" + dataset_name +"}"] + [""] * 15])
            for category in categories:
                row = []
                fus_woatt_scores, feb_scores, fus_watt_scores = self.get_scores_for_category(dataset_type, category, metric)
                row.append("\\textit{" + category.replace("_", "\_") +"}")
                for method, spaces in self.METHOD_CONFIG:
                    method_scores_aligned = []

                    if "FuS" in spaces:
                        method_scores_aligned.append(fus_woatt_scores.get(method, "NA"))

                    if "FeB" in spaces:
                        method_scores_aligned.append(feb_scores.get(method, "NA"))

                    if "FuS+A" in spaces:
                        method_scores_aligned.append(fus_watt_scores.get(method, "NA"))

                    ### HIGHLIGHTING BEST SCORES
                    valid_scores = [s for s in method_scores_aligned if s != "NA"]
                    if len(valid_scores) > 1:
                        max_score = max(valid_scores)
                        for i in range(len(method_scores_aligned)):
                            if method_scores_aligned[i] == max_score:
                                method_scores_aligned[i] = f"\\textbf{{{method_scores_aligned[i]}}}"

                    row.extend(method_scores_aligned)

                ###HIGHLIGHT ROW MAX
                row = self.underline_max_value(row)

                data.append(row)

        df = pd.DataFrame(data, columns=columns, index=None)
        print_latex_table(df.to_latex(index=False), metric=metric)



    def get_scores_for_category(self, dataset_type, category, metric):
        fs_woatt_scores = self.get_fs_baseline_scores(dataset_type, category, metric, att=False)
        ens_scores = self.get_ens_baseline_scores(dataset_type, category, metric)
        fs_watt_scores = self.get_fs_baseline_scores(dataset_type, category, metric, att=True)

        return fs_woatt_scores, ens_scores, fs_watt_scores



    def extract_method_name(self, to_extract_from, to_search_in):
        for method in to_extract_from:
            if method.lower() in to_search_in.lower():
                return method
        return None