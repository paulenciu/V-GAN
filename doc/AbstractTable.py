import ast
import re
from abc import ABC, abstractmethod

from src.data.dataset_type import DatasetType
from src.run.pipeline.BaselinePipeline import mvtec_categories


class AbstractTable(ABC):

    @abstractmethod
    def generate_table(self, metric="auc"):
        pass


    def is_fs_model(self, row):
        od_model = row["OD Method"]
        od_dict = ast.literal_eval(od_model)
        return od_dict["Ensemble Description"]["ensemble weight"] == 1.0

    def extract_name_and_score_from_row(self, row, metric):
        od_model = row["OD Method"]
        od_dict = ast.literal_eval(od_model)
        return od_dict["Ensemble Description"]["Ensemble Model"], row[metric.upper()]

    def underline_max_value(self, row):
        numeric_values = []
        for val in row[1:]:  # skip label at index 0
            match = re.search(r'(\d+\.\d+)', val)
            numeric_values.append(float(match.group(1)) if match else float('-inf'))

        max_val = max(numeric_values)

        for i in range(1, len(row)):
            if re.search(rf'{max_val:.3f}', row[i]):
                if '\\textbf{' in row[i]:
                    content = re.search(r'\\textbf{(.*?)}', row[i]).group(1)
                    row[i] = f"\\underline{{\\textbf{{{content}}}}}"
                else:
                    row[i] = f"\\underline{{{row[i]}}}"
        return row
