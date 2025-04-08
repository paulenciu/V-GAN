import os
import ast
from pathlib import Path

import pandas as pd
from av.codec import codec_descriptor

from src.data import dataset_type
from src.data.dataset_type import DatasetType
from src.run.pipeline.BaselinePipeline import mvtec_categories, cifar10_classes, fashionmnist_categories


def print_latex_code(df, dataset_name, metric):
    formatted_df = df.copy()

    # Define column ranges for each section (excluding the first 'Dataset' column)
    sections = {
        'Full Space': slice(1, 7),  # Columns 1-6 (0-indexed)
        'FeatureBagging': [7],  # Column 7
        'VGAN': slice(8, 11)  # Columns 8-10
    }

    # Process each row
    for row_idx, row in formatted_df.iterrows():
        # Collect all comparable values from all sections
        all_values = []
        for cols in sections.values():
            section_values = [
                float(x) if str(x).replace('.', '').replace('NA', '').isdigit() else -float('inf')
                for x in row.iloc[cols]
            ]
            all_values.extend(section_values)

        # Find global max for the row (ignore -inf)
        valid_values = [v for v in all_values if v != -float('inf')]
        if not valid_values:
            continue

        global_max = max(valid_values)

        # Bold the max value(s) wherever they appear in the row
        for col_idx, val in enumerate(row[1:]):  # Skip first column (dataset name)
            try:
                # Handle NA values
                if str(val) == 'NA':
                    continue

                num_val = float(val)
                if abs(num_val - global_max) < 1e-6:
                    # Double escape backslashes for LaTeX
                    formatted_df.iloc[row_idx, col_idx + 1] = r"\textbf{" + f"{num_val:.3f}" + "}"
            except (ValueError, TypeError):
                pass
    latex_code = formatted_df.to_latex(
        index=False,
        caption=f"{metric.upper()} scores",
        label=f"tab:{metric.lower()}_scores_{dataset_name.lower()}",
        escape=False,
        column_format="l|cccccc|c|ccc"
    )

    latex_code = latex_code.replace(
        "\multicolumn{6}{r}{Full Space} & FeatureBagging & \multicolumn{3}{r}{VGAN}",
        r"\multicolumn{6}{c|}{Full Space} & \multicolumn{1}{c|}{FeatureBagging} & \multicolumn{3}{c}{\VGANV}"
    ).replace(
        "LUNAR & LOF & KNN & PADIM & DFM & STFPM & LUNAR & LUNAR & LOF & KNN",
        r"LUNAR & LOF & KNN & PADIM & DFM & STFPM & LUNAR & LUNAR & LOF & KNN"
    )
    #.replace(
    #     "\\begin{table}",
    #     "\\begin{longtable}"
    # ).replace(
    #     "\\end{table}",
    #     "\\end{longtable}"
    # )

    latex_code = latex_code.replace(r"\midrule", r"\cline{0-7} \cline{8-8} \cline{9-11}")

    print(latex_code)


def get_full_space_scores(dataset_type, category, metric):
    fs_methods = ["LUNAR", "LOF", "KNN"]
    root_dir = Path("../experiments/od_baselines") / dataset_type.name / category
    anomalib_benchmark_file = Path("../experiments/od_baselines") / dataset_type.name / "anomaly_benchmarks.csv"

    od_scores = []
    # GETTING FS VALUES OF LUNAR, LOF, and KNN
    for fname in os.listdir(root_dir):
        if any([fname.startswith(prefix) for prefix in fs_methods]):
            score_df = pd.read_csv(root_dir / fname)
            score = score_df[metric.upper()].values[0]
            od_scores.append(f"{score:.3f}")

    # GETTING anomalib scores
    fs_methods = ["PADIM", "DFM", "STFPM"]
    if metric == "auc":
        anomalib_df = pd.read_csv(anomalib_benchmark_file)
        category_rows = anomalib_df[
            anomalib_df["category"].str.lower() == category.lower()
        ]
        for method in fs_methods:
            model_row = category_rows[category_rows["model"].str.lower() == method.lower()]
            score = model_row["auroc"].values[0]
            od_scores.append(f"{score:.3f}")
    else:
        od_scores.extend(["NA"] * len(fs_methods))

    return od_scores

def get_ens_scores(dataset_type, category, metric):
    root_dir = Path("../experiments/od_baselines") / dataset_type.name / category
    for fname in os.listdir(root_dir):
        if fname.startswith("FeatureBagging"):
            score_df = pd.read_csv(root_dir / fname)
            return [f"{score_df[metric.upper()].values[0]:.3f}"]

    return ["NA"]


def is_fs_model(row):
    od_model = row["OD Method"]
    od_dict = ast.literal_eval(od_model)
    return  od_dict["Ensemble Description"]["ensemble weight"] == 1.0


def extract_name_and_score_from_row(row, metric):
    od_model = row["OD Method"]
    od_dict = ast.literal_eval(od_model)
    return od_dict["Ensemble Description"]["Ensemble Model"], row[metric.upper()]


def get_vgan_scores(dataset_type, category, metric, date="21-03"):

    vgan_scores = {"LUNAR": "NA", "LOF": "NA", "KNN": "NA"}

    root_dir = Path("../experiments/remote/") / date
    prefix = dataset_type.name + "[" + category.split("/")[0]
    for fname in os.listdir(root_dir):
        if fname.startswith(prefix):
            score_df = pd.read_csv(root_dir / fname / "od_stats_-1.csv")
            for _, row in score_df.iterrows():
                if is_fs_model(row):
                    name, score = extract_name_and_score_from_row(row, metric)
                    score = f"{score:.3f}"
                    vgan_scores[name] = score
    return vgan_scores.values()

def generate_latex_tables(metric="auc"):
    config = [
        (DatasetType.MVTEC_AD, "MVTec AD", mvtec_categories),
        (DatasetType.OCCCIFAR10, "Cifar10", cifar10_classes),
        (DatasetType.OCCFMNIST, "FashionMNIST", fashionmnist_categories)
    ]

    for dataset_type, dataset_name, categories in config:
        columns = pd.MultiIndex.from_tuples([
            (dataset_name, ""),
            ("Full Space", "LUNAR"),
            ("Full Space", "LOF"),
            ("Full Space", "KNN"),
            ("Full Space", "PADIM"),
            ("Full Space", "DFM"),
            ("Full Space", "STFPM"),
            ("FeatureBagging", "LUNAR"),
            ("VGAN", "LUNAR"),
            ("VGAN", "LOF"),
            ("VGAN", "KNN"),
        ])

        data = []
        for category in categories:
            row = []
            row.append(category)
            fs_scores = get_full_space_scores(dataset_type, category, metric)
            ens_scores = get_ens_scores(dataset_type, category, metric)
            vgan_scores = get_vgan_scores(dataset_type, category, metric)
            row.extend(fs_scores)
            row.extend(ens_scores)
            row.extend(vgan_scores)

            data.append(row)


        df = pd.DataFrame(data, columns=columns)
        print_latex_code(df, dataset_name, metric)



