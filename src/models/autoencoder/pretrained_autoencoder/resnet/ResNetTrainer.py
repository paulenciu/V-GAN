import torch

from src.data.dataset.PreEmbeddedDataset import PreEmbeddedDataset
from src.data.dataset_loader import load_data
from src.data.dataset_type import DatasetType
from src.models.autoencoder.pretrained_autoencoder.resnet.pytorch.PyTorchResNet50AutoEncoder import \
    PyTorchResNet50AutoEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d, unflatten_images_3d
from matplotlib import pyplot as plt


class ResNetTrainer:

    def __init__(self, encoder, decoder, dataset, epochs=10000, batch_size=200, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_channels, height, width = dataset.image_shape
        flattened_images = extract_and_flatten_images_dataset_3d(dataset).cpu()
        unflattened_images = unflatten_images_3d(flattened_images, n_channels, height, width)
        self.dataset = unflattened_images
        self.train_dataset = PreEmbeddedDataset(unflattened_images, encoder, self.device)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr


    def train(self):
        data_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)
        loss_function = torch.nn.MSELoss()
        for epoch in range(self.epochs):
            print(f'\rEpoch {epoch} of {self.epochs}')
            for batch, embeddings in tqdm(data_loader, leave=False):
                optimizer.zero_grad()

                embeddings = embeddings.view(-1, 2048, 7, 7).to(self.device)
                batch = batch.to(self.device)

                reconstructions = self.decoder(embeddings)
                loss = loss_function(reconstructions, batch)

                loss.backward()
                optimizer.step()

            if epoch % 100 == 0:
                self.sample_plot()

    def sample_plot(self):
        sample = next(iter(DataLoader(dataset=self.train_dataset, batch_size=5)))
        fig, ax = plt.subplots(nrows=5, ncols=2)

        embeddings = sample[1].view(-1, 2048, 7, 7).to(self.device)
        reconstructions = self.decoder(embeddings)

        for i in range(5):
            ax[i, 0].imshow(reconstructions[i].detach().cpu())
            ax[i, 1].imshow(embeddings[i].detach().cpu())
