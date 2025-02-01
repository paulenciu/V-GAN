import torchvision.datasets

from src.data.IDataset import IDataset


class FashionMNISTDataset(IDataset):

    def __init__(self, root, category=None, train=True, transform=torchvision.transforms.ToTensor(), download=True):

        if category is None:
            category = []

        dataset = torchvision.datasets.FashionMNIST(root, train=train, transform=transform, download=download)

        category_dataset = [(img, label) for img, label in dataset if str(label) in category]
        data, labels = zip(*category_dataset)

        img_shape = category_dataset[0][0].shape

        super().__init__(img_shape, data, labels)
