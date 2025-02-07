import torchvision
from torchvision.transforms import transforms

from src.data.dataset.occ.OCCDataset import OCCDataset


class OCCFMNIST(OCCDataset):

    def fetch_dataset(self, root_dir, train, transform, download, normalize=False):

        if normalize:
            transform = transforms.Compose([
                transform,
                transforms.Normalize(mean=[0.2860], std=[0.3530])
            ])

        return torchvision.datasets.FashionMNIST(root_dir, train=train, transform=transform, download=download)
