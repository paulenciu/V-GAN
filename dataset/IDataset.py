import torch


class IDataset(torch.utils.data.Dataset):

    def __init__(self, image_size):
        self.image_size = image_size