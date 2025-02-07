import torchvision
from torchvision.transforms import transforms

from src.data.dataset.occ.OCCDataset import OCCDataset


class OCCCifar10(OCCDataset):
    """
    Dataset constellation:
    train dataset: 5000 samples inlier
    test_dataset: 10000 samples, of which 10% (=1000) are inlier
    """
    def fetch_dataset(self, root_dir, train, transform, download, normalize=False):

        if normalize:
            transform = transforms.Compose([
                transform,
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
            ])

        return torchvision.datasets.CIFAR10(
            root=root_dir,
            train=train,
            transform=transform,
            download=download)