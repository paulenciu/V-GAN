import torchvision
import random

from IPython.lib.deepreload import original_import

from src.data.IDataset import IDataset


class Cifar10Dataset(IDataset):

    def __init__(self, root_dir, inlier_category=None, train=True, transform=torchvision.transforms.ToTensor(), download=True):

        if inlier_category is None:
            inlier_category = []

        dataset = torchvision.datasets.CIFAR10(root=root_dir, train=train, transform=transform, download=download)

        combined_category = list(inlier_category)

        if not train:
            all_category = [str(c) for c in range(len(dataset.classes) - 1)]
            all_category_train_category_excluded = list(filter(lambda c: c not in inlier_category, all_category))
            sampled_train_category_excluded = random.sample(all_category_train_category_excluded, 2)
            combined_category += sampled_train_category_excluded

        category_dataset = [(img, label) for (img, label) in dataset if str(label) in combined_category]

        # BINARY CLASSIFIER
        # 0 = Inlier, 1 = Outlier
        if not train:
            category_dataset = list(map(lambda x: (x[0], 0) if str(x[1]) in inlier_category else (x[0], 1), category_dataset))

        data, labels = zip(*category_dataset)

        image_shape = dataset[0][0].shape

        super().__init__(image_shape, data, labels)


