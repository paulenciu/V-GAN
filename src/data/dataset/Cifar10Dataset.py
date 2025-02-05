import torchvision
import random
from src.data.IDataset import IDataset


class Cifar10Dataset(IDataset):

    def __init__(self, root_dir, category=None, train=True, transform=torchvision.transforms.ToTensor(), download=True):
        if category is None:
            category = []

        dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=download)

        if not train:
            all_category = [str(c) for c in range(len(dataset.classes) - 1)]
            all_category_train_category_excluded = list(filter(lambda c: c not in category, all_category))
            sampled_train_category_excluded = random.sample(all_category_train_category_excluded, 2)
            category += sampled_train_category_excluded

        category_dataset = [(img, label) for (img, label) in dataset if str(label) in category]

        data, labels = zip(*category_dataset)

        image_shape = dataset[0][0].shape

        super().__init__(image_shape, data, labels)


