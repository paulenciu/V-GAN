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


def print_latex_table(latex_code, model, metric="auc"):
    modified_code = (
        "\\begin{table*}\n"
        "\\centering\n"
        "\\caption{Encoded Space - " + model.upper().replace("_", "\_") + " " + metric.upper() + " Score}\n"
        "\\label{pixel_space" + metric.lower() + "}\n"
        + latex_code +
        "\\end{table*}"
    )

    modified_code = modified_code.replace(
        r'\begin{tabular}{llllllllll}',
        r'\begin{tabular}{l|ccc|ccc|ccc}'
    )

    # Center method headers
    for method in ["LUNAR", "LOF", "KNN"]:
        modified_code = modified_code.replace(
            rf'\multicolumn{{3}}{{r}}{{{method}}}',
            rf'\multicolumn{{3}}{{c}}{{{method}}}'
        )

    print(modified_code)



class EncSpaceTable(AbstractTable):

    DATASET_CONFIG = [
        (DatasetType.MVTEC_AD, "MVTec AD", mvtec_categories),
        #(DatasetType.OCCCIFAR10, "Cifar10", cifar10_classes),
        #(DatasetType.OCCFMNIST, "FashionMNIST", fashionmnist_categories)
    ]

    METHOD_CONFIG = [
        ("LUNAR", ["FuS", "FeB", "VGAN"]),
        ("LOF", ["FuS", "FeB", "VGAN"]),
        ("KNN", ["FuS", "FeB", "VGAN"]),
    ]

    def get_fs_baseline_scores(self, dataset_type, category, metric, model):

        # GETTING FS VALUES OF LUNAR, LOF, and KNN
        fs_methods = ["LUNAR", "LOF", "KNN"]
        root_dir = Path("../experiments/od_baselines/embeddingspace") / model / dataset_type.name / category

        scores = {}
        for fname in os.listdir(root_dir):
            if any([fname.startswith(prefix) for prefix in fs_methods]):
                method_name = self.extract_method_name(to_extract_from=fs_methods, to_search_in=fname)
                score_df = pd.read_csv(root_dir / fname)
                score = score_df[metric.upper()].values[0]
                scores[method_name] = (f"{score:.3f}")

        return scores

    def get_ens_baseline_scores(self, dataset_type, category, metric, model):
        root_dir = Path("../experiments/od_baselines/embeddingspace") / model / dataset_type.name / category
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

    def get_vgan_pixel_scores(self, dataset_type, category, metric, date="09-04", model="L_16_VIT"):
        methods = ["LUNAR", "LOF", "KNN"]
        vgan_scores = {k: "NA" for k in methods}
        suffix = ""
        if model == "resnet18":
            suffix = "_r18"
        elif model == "resnet50":
            suffix = "_r50"

        root_dir = Path("../experiments/remote/") / date
        prefix = dataset_type.name + "[" + category.split("/")[0]
        for fname in os.listdir(root_dir):
            if fname.__contains__(prefix) and fname.endswith(suffix):
                score_df = pd.read_csv(root_dir / fname / "od_stats_-1.csv")
                for _, row in score_df.iterrows():
                    if self.is_fs_model(row):
                        name, score = self.extract_name_and_score_from_row(row, metric)
                        score = f"{score:.3f}"
                        vgan_scores[name] = score
        return vgan_scores

    def generate_table(self, metric="auc", model="L_16_VIT", date="09-04"):

        cols = [("", "")]

        for method, spaces in self.METHOD_CONFIG:
            for space in spaces:
                cols.append((method, space))


        columns = pd.MultiIndex.from_tuples(cols)
        data = []
        for dataset_type, dataset_name, categories in self.DATASET_CONFIG:
            data.extend([["\\textbf{" + dataset_name +"}"] + [""] * 9])
            for category in categories:
                row = []
                fus_scores, feb_scores, vgan_scores = self.get_scores_for_category(dataset_type, category, metric, model, date)
                row.append("\\textit{" + category.replace("_", "\_") +"}")
                for method, spaces in self.METHOD_CONFIG:
                    method_scores_aligned = []

                    if "FuS" in spaces:
                        method_scores_aligned.append(fus_scores.get(method, "NA"))

                    if "FeB" in spaces:
                        method_scores_aligned.append(feb_scores.get(method, "NA"))

                    if "VGAN" in spaces:
                        method_scores_aligned.append(vgan_scores.get(method, "NA"))


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
        print_latex_table(df.to_latex(index=False), metric=metric, model=model)


    def get_scores_for_category(self, dataset_type, category, metric, model, date):
        fs_scores = self.get_fs_baseline_scores(dataset_type, category, metric, model)
        ens_scores = self.get_ens_baseline_scores(dataset_type, category, metric, model)
        vgan_scores = self.get_vgan_pixel_scores(dataset_type, category, metric, date=date, model=model)

        return fs_scores, ens_scores, vgan_scores


    def extract_method_name(self, to_extract_from, to_search_in):
        for method in to_extract_from:
            if method.lower() in to_search_in.lower():
                return method
        return None