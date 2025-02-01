from enum import Enum
from pathlib import Path

import torchvision

from src.data.dataset.Cifar10Dataset import Cifar10Dataset
from src.data.IDataset import IDataset
from src.data.dataset.FashionMNISTDataset import FashionMNISTDataset
from src.data.dataset.MVTecADDataset import MVTecADDataset
from src.data.dataset.SyntheticImageDataset import SyntheticImageDataset


class DatasetType(Enum):

    def __init__(self, dataset_class, file_path: str):
        self.dataset_class = dataset_class  # Constructor for the data class
        self.file_path = Path(file_path)


    # Define enum members
    CIFAR10 = Cifar10Dataset, "../datasets/cifar10"
    FASHION_MNIST = FashionMNISTDataset, "../datasets/fashion_mnist"
    MVTEC_AD = MVTecADDataset, "../datasets/mvtec_ad"
    SYNTHETIC = SyntheticImageDataset, ""

    def load(self, category, train=True, transform=torchvision.transforms.ToTensor()):
        """
        Load the data using the constructor and file path.
        """
        return self.dataset_class(self.file_path, category=category, train=train, transform=transform)
