from enum import Enum
from pathlib import Path

import torchvision

from src.data.dataset.Cifar10Dataset import Cifar10Dataset
from src.data.dataset.FashionMNISTDataset import FashionMNISTDataset
from src.data.dataset.MVTecADDataset import MVTecADDataset
from src.data.dataset.occ.OCCCifar10 import OCCCifar10
from src.data.dataset.occ.OCCCifar100 import OCCCifar100
from src.data.dataset.occ.OCCDataset import OCCDataset
from src.data.dataset.SyntheticImageDataset import SyntheticImageDataset
from src.data.dataset.occ.OCCFMNIST import OCCFMNIST
from src.data.dataset.occ.OCCMNIST import OCCMNIST


class DatasetType(Enum):

    def __init__(self, dataset_class, file_path: str):
        self.dataset_class = dataset_class  # Constructor for the data class
        self.file_path = Path(file_path)


    # Define enum members
    CIFAR10 = Cifar10Dataset, "../datasets/cifar10"
    OCCCIFAR10 = OCCCifar10, "../datasets/cifar10"
    OCCCIFAR100 = OCCCIFAR100, "../datasets/cifar100"
    OCCFMNIST = OCCFMNIST, "../datasets/fashion_mnist"
    OCCMNIST = OCCMNIST, "../datasets/mnist"
    FASHION_MNIST = FashionMNISTDataset, "../datasets/fashion_mnist"
    MVTEC_AD = MVTecADDataset, "../datasets/mvtec_ad"
    SYNTHETIC = SyntheticImageDataset, ""

    def load(self, category, train=True, transform=torchvision.transforms.ToTensor(), normalize=False):
        """
        Load the data using the constructor and file path.
        """
        return self.dataset_class(self.file_path, inlier_category=category, train=train, transform=transform, normalize=normalize)
