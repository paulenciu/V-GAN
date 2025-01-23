import torchvision.datasets

from src.dataset.IDataset import IDataset


class FashionMNISTTrousersDataset(IDataset):

    def __init__(self, root, train=True, transform=torchvision.transforms.ToTensor(), download=True):
        dataset = torchvision.datasets.FashionMNIST(root, train=train, transform=transform, download=download)
        trousers_dataset = [(img, label) for img, label in dataset if label == 1]

        img_shape = trousers_dataset[0][0].shape

        super().__init__(img_shape, trousers_dataset)
