import torchvision

from src.data.dataset.occ.OCCDataset import OCCDataset
from torchvision import datasets



class OCCMNIST(OCCDataset):


    def fetch_dataset(self, root_dir, train, download, transform):
       return datasets.MNIST(root=root_dir, train=train, download=download, transform=transform)