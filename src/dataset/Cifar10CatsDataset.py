import torchvision
from src.dataset.IDataset import IDataset


class Cifar10CatsDataset(IDataset):

    def __init__(self, root, train=True, transform=torchvision.transforms.ToTensor(), download=True):
        dataset = torchvision.datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
        cats_dataset = [(img, label) for (img, label) in dataset if label == 3]

        image_shape = dataset[0][0].shape

        super().__init__(image_shape, cats_dataset)


