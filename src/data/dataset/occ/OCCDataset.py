from abc import ABC, abstractmethod

import numpy as np
import torchvision

from src.data.IDataset import IDataset


class OCCDataset(ABC, IDataset):

    def __init__(self, root_dir, inlier_category=None, train=True, transform=torchvision.transforms.ToTensor(), download=True):
        if inlier_category is None:
            inlier_category = []

        # FIXME change logic for test
        dataset = self.fetch_dataset(root_dir=root_dir, train=train, transform=transform, download=download)

        class_names = dataset.classes
        inlier_class = class_names.index(inlier_category)

        train_indices = np.where(np.array(dataset.targets) == inlier_class)[0]
        np.random.shuffle(train_indices)
        split = int(0.8 * len(train_indices))

        if train:
            train_indices = train_indices[:split]
            inlier_dataset = [(dataset[i][0], 0) for i in train_indices]
            data, labels = zip(*inlier_dataset)
        else:
            test_indices_outlier = np.where(np.array(dataset.targets) != inlier_class)[0]
            test_extra_indices = train_indices[split:]
            test_indices = np.concatenate([test_extra_indices, test_indices_outlier])
            data = [dataset[i][0] for i in test_indices]
            labels = [0 if dataset[i][1] == inlier_class else 1 for i in test_indices]

        image_shape = dataset[0][0].shape
        super().__init__(image_shape, data, labels)

    @abstractmethod
    def fetch_dataset(self, root_dir, train, transform, download):
        pass