from abc import ABC, abstractmethod

from src.data.IDataset import IDataset


class IVMMDLogger(ABC):

    @abstractmethod
    def log(self, data: IDataset, epoch=0):
        pass