import ast
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
