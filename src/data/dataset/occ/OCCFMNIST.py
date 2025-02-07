import torchvision

from src.data.dataset.occ.OCCDataset import OCCDataset


class OCCFMNIST(OCCDataset):

    def fetch_dataset(self, root_dir, train, transform, download):
        return torchvision.datasets.FashionMNIST(root_dir, train=train, transform=transform, download=download)
