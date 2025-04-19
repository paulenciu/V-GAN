import os
import ast
from collections import defaultdict
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from doc.AbstractTable import AbstractTable
from src.data import dataset_type
from src.data.dataset_type import DatasetType
from src.run.pipeline.BaselinePipeline import mvtec_categories, cifar10_classes, fashionmnist_categories


import pandas as pd
from scipy.stats import kruskal
from scikit_posthocs import posthoc_conover

def kruskal_conover_latex(df, alpha_1=0.10, alpha_2=0.05):
    """
    df: wide DataFrame. Columns = [LUNAR-FuS, LUNAR-FeB, LUNAR-VGAN, LOF-FuS, etc.]
        Rows = categories or datasets.
    alpha_1 = 0.10 threshold for single-symbol significance
    alpha_2 = 0.05 threshold for double-symbol significance

    Prints a LaTeX table showing pairwise Conover-Iman comparisons (+, -, etc.)
    """
    all_columns = df.columns.tolist()
    all_groups = []

    for row_method, _ in ConoverImanTable.METHOD_CONFIG:
        group_cols = [(m_name, s_name) for m_name, s_name in all_columns if m_name == row_method]
        all_groups.append(group_cols)

    for method_groups in all_groups:
        group_scores_per_method = [df[group].dropna().tolist() for group in method_groups]
        kw_stat, kw_p = kruskal(*group_scores_per_method)
        print(f"Kruskal-Wallis H={kw_stat:.3f}, p={kw_p:.3f}")

    # kw_stat, kw_p = kruskal(*all_columns)
    # print(f"Kruskal-Wallis H={kw_stat:.3f}, p={kw_p:.3f}")

    # # If KW is not significant at alpha_1 => Entire table is "grayed out"
    # if kw_p > alpha_1:
    #     print("Kruskal-Wallis not significant => Gray out entire table.")
    #
    #     latex_str = "\\begin{tabular}{l" + "c" * len(methods) + "}\n"
    #     latex_str += " & " + " & ".join(methods) + "\\\\ \\hline\n"
    #     for row_m in methods:
    #         row_str = row_m + " & " + " & ".join(["\\cellcolor{gray!30}--" for _ in methods])
    #         latex_str += row_str + "\\\\\n"
    #     latex_str += "\\end{tabular}\n"
    #
    #     print(latex_str)
    #     return
    #
    # # 2) Long-format for posthoc_conover
    # df_long = df.melt(var_name='Method', value_name='Score').dropna()
    #
    # # 3) Run Conover-Iman post-hoc test (Holm correction)
    # posthoc_res = posthoc_conover(df_long, val_col='Score', group_col='Method', p_adjust='holm')
    #
    # # 4) Compute ranks across entire data (non-parametric approach)
    # #    Rank all numeric values, ignoring group membership, then compute average rank per method.
    # df_long['GlobalRank'] = df_long['Score'].rank(method='average')
    # mean_ranks = df_long.groupby('Method')['GlobalRank'].mean()

    # 5) Build pairwise significance symbols
    # def significance_symbol(p_val, rank_row, rank_col):
    #     """
    #     Returns +, ++ if row < col (better), - or -- if row > col (worse), or blank otherwise.
    #     """
    #     if p_val <= alpha_2:
    #         symbol = "++" if rank_row < rank_col else "--"
    #     elif p_val <= alpha_1:
    #         symbol = "+" if rank_row < rank_col else "-"
    #     else:
    #         symbol = ""
    #     return symbol
    #
    # # Make a table of symbols
    # symbol_matrix = pd.DataFrame(index=methods, columns=methods, data="", dtype=object)
    # for m1 in methods:
    #     for m2 in methods:
    #         if m1 == m2:
    #             symbol_matrix.loc[m1, m2] = "—"  # Em dash on the diagonal
    #         else:
    #             p_val = posthoc_res.loc[m1, m2]
    #             symbol_matrix.loc[m1, m2] = significance_symbol(p_val, mean_ranks[m1], mean_ranks[m2])
    #
    # # 6) Convert to LaTeX
    # latex_str = "\\begin{tabular}{l" + "c" * len(methods) + "}\n"
    # latex_str += " & " + " & ".join(methods) + "\\\\ \\hline\n"
    # for row_m in methods:
    #     row_vals = [symbol_matrix.loc[row_m, col_m] for col_m in methods]
    #     row_str = row_m + " & " + " & ".join(row_vals)
    #     latex_str += row_str + "\\\\\n"
    # latex_str += "\\end{tabular}\n"
    #
    # print(latex_str)

class ConoverImanTable(AbstractTable):

    DATASET_CONFIG = [
        (DatasetType.MVTEC_AD, "MVTec AD", mvtec_categories),
        (DatasetType.OCCCIFAR10, "Cifar10", cifar10_classes),
        (DatasetType.OCCFMNIST, "FashionMNIST", fashionmnist_categories)
    ]

    METHOD_CONFIG = [
        ("LUNAR", ["FuS", "FeB", "VGAN"]),
        ("LOF", ["FuS", "FeB", "VGAN"]),
        ("KNN", ["FuS", "FeB", "VGAN"]),
    ]

    def get_fs_baseline_scores(self, dataset_type, category, metric, normalized):

        # GETTING FS VALUES OF LUNAR, LOF, and KNN
        fs_methods = ["LUNAR", "LOF", "KNN"]

        normalization = "normalized" if normalized else "unnormalized"

        root_dir = Path("../experiments/od_baselines/pixelspace") / normalization / dataset_type.name / category
        anomalib_benchmark_file = Path("../experiments/od_baselines/pixelspace") / dataset_type.name / "anomaly_benchmarks.csv"

        scores = {}
        for fname in os.listdir(root_dir):
            if any([fname.startswith(prefix) for prefix in fs_methods]):
                method_name = self.extract_method_name(to_extract_from=fs_methods, to_search_in=fname)
                score_df = pd.read_csv(root_dir / fname)
                score = score_df[metric.upper()].values[0]
                scores[method_name] = score

        # GETTING anomalib scores
        # fs_methods = ["PADIM", "DFM", "STFPM"]
        # if metric == "auc":
        #     anomalib_df = pd.read_csv(anomalib_benchmark_file)
        #     category_rows = anomalib_df[
        #         anomalib_df["category"].str.lower() == category.lower()
        #         ]
        #     for method_name in fs_methods:
        #         model_row = category_rows[category_rows["model"].str.lower() == method_name.lower()]
        #         score = model_row["auroc"].values[0]
        #         scores[method_name] = score
        # else:
        #     for method_name in fs_methods:
        #         scores[method_name] = np.nan

        return scores

    def get_ens_baseline_scores(self, dataset_type, category, metric, normalized):

        normalization = "normalized" if normalized else "unnormalized"
        root_dir = Path("../experiments/od_baselines/pixelspace") / normalization / dataset_type.name / category
        methods = ["LUNAR", "LOF", "KNN"]
        scores = {}
        for fname in os.listdir(root_dir):
            if fname.startswith("FeatureBagging"):
                score_df = pd.read_csv(root_dir / fname)

                for method_name in methods:
                    row = score_df[score_df["OD Method"].str.lower() == method_name.lower()]
                    scores[method_name]= row[metric.upper()].values[0]

                return scores

        return {k: np.nan for k in methods}

    def get_vgan_pixel_scores(self, dataset_type, category, metric,  model="", date="21-03"):
        methods = ["LUNAR", "LOF", "KNN"]
        vgan_scores = {k: np.nan for k in methods}

        root_dir = Path("../experiments/remote/") / date


        prefix = dataset_type.name + "[" + category.split("/")[0]

        if model != "":
            prefix = "embedding_" + prefix

        for fname in os.listdir(root_dir):
            if fname.startswith(prefix) and fname.endswith(model):
                score_df = pd.read_csv(root_dir / fname / "od_stats_-1.csv")
                for _, row in score_df.iterrows():
                    if self.is_fs_model(row):
                        name, score = self.extract_name_and_score_from_row(row, metric)
                        vgan_scores[name] = score
        return vgan_scores

    def generate_table(self, metric="auc", model="", date="21-03", normalized=True):

        cols = []

        for method, spaces in self.METHOD_CONFIG:
            for space in spaces:
                cols.append((method, space))


        columns = pd.MultiIndex.from_tuples(cols)
        data = []
        for dataset_type, dataset_name, categories in self.DATASET_CONFIG:
            #data.extend([["\\textbf{" + dataset_name +"}"] + [""] * 12])
            for category in categories:
                row = []
                fus_scores, feb_scores, vgan_scores = self.get_scores_for_category(dataset_type, category, metric, model, date, normalized)
                #row.append("\\textit{" + category +"}")
                for method, spaces in self.METHOD_CONFIG:
                    method_scores_aligned = []

                    if "FuS" in spaces:
                        method_scores_aligned.append(fus_scores.get(method, np.nan))

                    if "FeB" in spaces:
                        method_scores_aligned.append(feb_scores.get(method, np.nan))

                    if "VGAN" in spaces:
                        method_scores_aligned.append(vgan_scores.get(method, np.nan))

                    row.extend(method_scores_aligned)

                data.append(row)

        df = pd.DataFrame(data, columns=columns, index=None)
        kruskal_conover_latex(df)


    def get_scores_for_category(self, dataset_type, category, metric, model, date, normalized):
        fs_scores = self.get_fs_baseline_scores(dataset_type, category, metric, normalized)
        ens_scores = self.get_ens_baseline_scores(dataset_type, category, metric, normalized)
        vgan_scores = self.get_vgan_pixel_scores(dataset_type, category, metric, model, date)

        return fs_scores, ens_scores, vgan_scores



    def extract_method_name(self, to_extract_from, to_search_in):
        for method in to_extract_from:
            if method.lower() in to_search_in.lower():
                return method
        return None