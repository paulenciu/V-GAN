from abc import ABC, abstractmethod
from pathlib import Path

from src.data.IDataset import IDataset


class IVMMDLogger(ABC):

    @abstractmethod
    def log(self, data: IDataset, epoch=0):
        pass

    @abstractmethod
    def set_base_dir(self, base_dir: Path):
        pass