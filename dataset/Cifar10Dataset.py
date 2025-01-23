import torchvision
from dataset.IDataset import IDataset
from torch.utils.data import Dataset


class Cifar10Dataset(IDataset):

    def __init__(self, root, train=True, transform=None, download=True):
        dataset = torchvision.datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
        image_shape = dataset[0][0].shape

        super().__init__(image_shape)
        self.dataset = dataset


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]