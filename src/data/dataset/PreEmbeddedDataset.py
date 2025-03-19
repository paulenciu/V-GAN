import torch
from src.data import IDataset
from torch.utils.data import Dataset, DataLoader


class PreEmbeddedDataset(Dataset):
    def __init__(self, dataset: IDataset, encoder, device):
        self.dataset = dataset
        self.device = device
        self.encoder = encoder.to(device)
        self.encoder.eval()

        with torch.no_grad():
            self.embeddings = torch.cat([
                self.encoder(batch.to(device)).view(batch.size(0), -1)
                for batch in DataLoader(dataset, batch_size=500)
            ], dim=0).cpu()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.embeddings[idx]

    def get_embedding(self, img):
        idx = self.dataset.dataset.index(img)
        return self.embeddings[idx]

    def is_embedding(self, element):
        return True if element in self.embeddings else False