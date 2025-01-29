import torchvision
from src.data.IDataset import IDataset


class Cifar10Dataset(IDataset):

    def __init__(self, root_dir, category=None, train=True, transform=torchvision.transforms.ToTensor(), download=True):
        if category is None:
            category = []

        dataset = torchvision.datasets.CIFAR10(root=root_dir, train=train, transform=transform, download=download)
        category_dataset = [(img, label) for (img, label) in dataset if label in category]

        data, labels = zip(*category_dataset)

        image_shape = dataset[0][0].shape

        super().__init__(image_shape, data, labels)


