from abc import ABC, abstractmethod
import numpy as np
import torch

class BaseOutlierDetector(ABC):

    @abstractmethod
    def fit(self, subspaces):
        pass

    @abstractmethod
    def decision_function(self, x):
        pass