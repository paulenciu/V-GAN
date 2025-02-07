import torchvision.datasets
import random

from torchvision.transforms import transforms

from src.data.IDataset import IDataset


class FashionMNISTDataset(IDataset):

    def __init__(self, root, category=None, train=True, transform=torchvision.transforms.ToTensor(), download=True, normalize=False):

        if category is None:
            category = []

        dataset = torchvision.datasets.FashionMNIST(root, train=train, transform=transform, download=download)

        if normalize:
            transform = transforms.Compose([
                transform,
                transforms.Normalize(mean=[0.2860], std=[0.3530])
            ])


        if not train:
            all_category = [str(c) for c in range(len(dataset.classes) - 1)]
            all_category_train_category_excluded = list(filter(lambda c: c not in category, all_category))
            sampled_train_category_excluded = random.sample(all_category_train_category_excluded, 1)
            category += sampled_train_category_excluded

        category_dataset = [(img, label) for (img, label) in dataset if str(label) in category]

        data, labels = zip(*category_dataset)

        image_shape = dataset[0][0].shape

        super().__init__(image_shape, data, labels)
