import torch


class IDataset(torch.utils.data.Dataset):

    def __init__(self, image_shape, dataset, labels):
        self.image_shape = image_shape
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index] if len(self.labels) > 0 else None