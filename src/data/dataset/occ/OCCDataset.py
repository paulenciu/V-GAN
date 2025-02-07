from abc import ABC, abstractmethod

import numpy as np
import torchvision

from src.data.IDataset import IDataset


class OCCDataset(ABC, IDataset):

    def __init__(self, root_dir, inlier_category=None, train=True, transform=torchvision.transforms.ToTensor(), download=True, normalize=False):
        if inlier_category is None:
            inlier_category = []

        dataset = self.fetch_dataset(root_dir=root_dir, train=train, transform=transform, download=download, normalize=normalize)

        class_names = dataset.classes
        inlier_class = class_names.index(inlier_category)

        # BINARY CLASSIFICATION
        # Label Inlier = 0; Outlier = 1
        if train:
            train_indices = np.where(np.array(dataset.targets) == inlier_class)[0]
            np.random.shuffle(train_indices)
            inlier_dataset = [(dataset[i][0], 0) for i in train_indices]
            data, labels = zip(*inlier_dataset)
        else:
            dataset = [(dataset[i][0], 0) if dataset[i][1] == inlier_class else (dataset[i][0], 1) for i in range(len(dataset))]
            data, labels = zip(*dataset)


        image_shape = dataset[0][0].shape
        super().__init__(image_shape, data, labels)

    @abstractmethod
    def fetch_dataset(self, root_dir, train, transform, download, normalize=False):
        pass