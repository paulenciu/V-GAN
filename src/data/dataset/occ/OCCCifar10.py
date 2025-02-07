import torchvision

from src.data.dataset.occ.OCCDataset import OCCDataset


class OCCCifar10(OCCDataset):

    def fetch_dataset(self, root_dir, train, transform, download):
        return torchvision.datasets.CIFAR10(root=root_dir, train=train, transform=transform, download=download)