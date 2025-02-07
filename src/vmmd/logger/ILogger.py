from abc import ABC, abstractmethod

from src.data.IDataset import IDataset


class ILogger(ABC):

    @abstractmethod
    def log(self, data: IDataset):
        pass