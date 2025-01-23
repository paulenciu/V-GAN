import torch


class IDataset(torch.utils.data.Dataset):

    def __init__(self, image_shape, dataset):
        self.image_shape = image_shape
        self.dataset = dataset

    def get_image_size(self):
        return self.image_shape[1], self.image_shape[2]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]