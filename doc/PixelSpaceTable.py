import os
import ast
from collections import defaultdict
from pathlib import Path

import pandas as pd

from doc.AbstractTable import AbstractTable
from src.data import dataset_type
from src.data.dataset_type import DatasetType
from src.run.pipeline.BaselinePipeline import mvtec_categories, cifar10_classes, fashionmnist_categories

class PixelSpaceTable(AbstractTable):

    DATASET_CONFIG = [
        (DatasetType.MVTEC_AD, "MVTec AD", mvtec_categories),
        (DatasetType.OCCCIFAR10, "Cifar10", cifar10_classes),
        (DatasetType.OCCFMNIST, "FashionMNIST", fashionmnist_categories)
    ]

    METHOD_CONFIG = ["LUNAR", "LOF", "KNN", "PADIM", "DFM", "STFPM"]

    def get_fs_baseline_scores(self, dataset_type, category, metric):

        # GETTING FS VALUES OF LUNAR, LOF, and KNN
        fs_methods = ["LUNAR", "LOF", "KNN"]
        root_dir = Path("../experiments/od_baselines/pixelspace") / dataset_type.name / category
        anomalib_benchmark_file = Path("../experiments/od_baselines") / dataset_type.name / "anomaly_benchmarks.csv"

        scores = {}
        for fname in os.listdir(root_dir):
            if any([fname.startswith(prefix) for prefix in fs_methods]):
                method_name = self.extract_method_name(to_extract_from=fs_methods, to_search_in=fname)
                score_df = pd.read_csv(root_dir / fname)
                score = score_df[metric.upper()].values[0]
                scores[method_name] = (f"{score:.3f}")

        # GETTING anomalib scores
        fs_methods = ["PADIM", "DFM", "STFPM"]
        if metric == "auc" and False:
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

    def get_vgan_pixel_scores(self, dataset_type, category, metric, date="21-03"):
        methods = ["LUNAR", "LOF", "KNN"]
        vgan_scores = {k: "NA" for k in methods}

        root_dir = Path("../experiments/remote/") / date
        prefix = dataset_type.name + "[" + category.split("/")[0]
        for fname in os.listdir(root_dir):
            if fname.startswith(prefix):
                score_df = pd.read_csv(root_dir / fname / "od_stats_-1.csv")
                for _, row in score_df.iterrows():
                    if self.is_fs_model(row):
                        name, score = self.extract_name_and_score_from_row(row, metric)
                        score = f"{score:.3f}"
                        vgan_scores[name] = score
        return vgan_scores

    # Split categories into groups for table sections
    CATEGORY_GROUPS = {
        DatasetType.MVTEC_AD.name: [mvtec_categories[:7], mvtec_categories[7:]],
        DatasetType.OCCCIFAR10.name: [cifar10_classes],
        DatasetType.OCCFMNIST.name: [fashionmnist_categories]
    }

    def generate_table(self, metric="auc"):
        latex_code = []
        latex_code.append(r"\begin{sidewaystable}")
        latex_code.append(r"\centering\small")
        latex_code.append(r"\caption{%s scores}" % metric.upper())
        latex_code.append(r"\label{tab:%s_scores}" % metric.lower())

        # Generate table for each dataset
        for dataset_type, dataset_name, _ in self.DATASET_CONFIG:
            groups = self.CATEGORY_GROUPS[dataset_type.name]

            for group_idx, category_group in enumerate(groups):
                latex_code.append(r"\begin{minipage}{\textwidth}")
                latex_code.append(r"\resizebox{\textwidth}{!}{")
                latex_code.append(r"\begin{tabular}{l" + "|ccc" * len(category_group) + "|}")

                # Header construction
                latex_code.append(r"\toprule")
                latex_code.append(r"\multirow{2}{*}{\rotatebox{90}{%s}} & \multicolumn{%d}{c}{%s} \\" % (
                    dataset_name,
                    len(category_group) * 3,
                    "Dataset Categories" if group_idx == 0 else "Continued Dataset Categories"
                ))

                # Category names
                category_header = " & ".join([r"\multicolumn{3}{c}{%s}" % cat for cat in category_group])
                latex_code.append(r"\cmidrule(lr){2-%d} %s \\" % (1 + len(category_group) * 3, category_header))

                # Metric columns
                metric_header = " & ".join([r"FuS & FeB & VGAN" for _ in category_group])
                latex_code.append(r"Method & " + metric_header + r" \\")
                latex_code.append(r"\midrule")

                # Data rows
                for method in self.METHOD_CONFIG:
                    row = [method]
                    for category in category_group:
                        scores = self.get_scores_for_category(dataset_type, category, method, metric)
                        row.extend([
                            scores.get("fs", "NA"),
                            scores.get("ens", "NA"),
                            scores.get("vgan", "NA")
                        ])
                    latex_code.append(" & ".join(map(str, row)) + r" \\")

                latex_code.append(r"\bottomrule")
                latex_code.append(r"\end{tabular}}")
                latex_code.append(r"\end{minipage}")
                latex_code.append(r"\vspace{0.5cm}")  # Space between table sections

        latex_code.append(r"\end{sidewaystable}")
        print("\n".join(latex_code))

    def get_scores_for_category(self, dataset_type, category, method, metric):
        # Implement your existing score collection logic here
        fs_scores = self.get_fs_baseline_scores(dataset_type, category, metric)
        ens_scores = self.get_ens_baseline_scores(dataset_type, category, metric)
        vgan_scores = self.get_vgan_pixel_scores(dataset_type, category, metric)

        return {
            "fs": self.format_score(fs_scores.get(method)),
            "ens": self.format_score(ens_scores.get(method)),
            "vgan": self.format_score(vgan_scores.get(method))
        }

    def format_score(self, score):
        try:
            return r"\textbf{%s}" % f"{float(score):.3f}" if float(
                score) == self.get_max_score() else f"{float(score):.3f}"
        except:
            return "NA"

    def print_latex_code(self, df, metric):
        latex_code = df.to_latex(
            index=False,
            caption=f"{metric.upper()} scores",
            label=f"tab:{metric.lower()}_scores",
            escape=False,
            column_format="l|" + "|".join(["ccc"] * sum(len(d[2]) for d in self.DATASET_CONFIG)) + "|",
            multicolumn_format="c",
        )

        # Add rotation and landscape formatting
        latex_code = latex_code.replace(
            r"\begin{tabular}",
            r"\begin{sidewaystable}\centering\small\begin{tabular}"
        ).replace(
            r"\end{tabular}",
            r"\end{tabular}\end{sidewaystable}"
        )

        # Fix multirow formatting
        current_pos = 1
        for dataset in self.DATASET_CONFIG:
            n_cols = len(dataset[2]) * 3
            latex_code = latex_code.replace(
                r"\toprule",
                f"\\multirow{{2}}{{*}}{{\\rotatebox{{90}}{{{dataset[1]}}}}}",
                1
            )
            current_pos += n_cols

        print(latex_code)

    def collect_score_data(self, metric):
        model_performance = {}

        for dataset_type, dataset_name, categories in self.DATASET_CONFIG:
            for category in categories:
                fs_scores = self.get_fs_baseline_scores(dataset_type, category, metric) or {}
                ens_scores = self.get_ens_baseline_scores(dataset_type, category, metric) or {}
                vgan_scores = self.get_vgan_pixel_scores(dataset_type, category, metric, date="21-03") or {}

                model_performance[category] = {
                    "fs": fs_scores,
                    "ens": ens_scores,
                    "vgan": vgan_scores,
                }
        return model_performance

    def extract_method_name(self, to_extract_from, to_search_in):
        for method in to_extract_from:
            if method.lower() in to_search_in.lower():
                return method
        return None