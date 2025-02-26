from abc import ABC, abstractmethod
import numpy as np
import torch

class BaseOutlierDetector(ABC):

    @abstractmethod
    def fit(self, subspaces, x):
        pass

    @abstractmethod
    def decision_score(self, x):
        pass