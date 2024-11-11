import datetime
import os
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torchvision

from models.Autoencoder import ResNet50Encoder
from vmmd import VMMD

def download_cifar10():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    return trainset, testset

class CatDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        trainset, _ = download_cifar10()
        cat_indices = [i for i, label in enumerate(trainset.targets) if label == 3]
        cat_subset = torch.utils.data.Subset(trainset, cat_indices)
        self.subset = cat_subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
def init_vmmd():
    vmmd = VMMD(batch_size=1, path_to_directory=Path(os.getcwd()) / "experiments" /
                 f"Example_normal_{datetime.datetime.now()}_vmmd", epochs=1500  )
    return vmmd

def init_encoder(name: str):
    if name == "resnet50":
        return ResNet50Encoder()
    elif name == "inceptionv4":
        return Inceptionv4(pretrained_model=timm.create_model('inception_v4', pretrained=True))
    else:
        raise ValueError("Encoder not found")

class Inceptionv4(nn.Module):
    def __init__(self, pretrained_model):
        super(Inceptionv4, self).__init__()

        self.upsample = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
        self.pretrained_model = pretrained_model
        self.pretrained_model.reset_classifier(0)  # 0 means no output classes
        self.pretrained_model.eval()

    def forward(self, x):
        x = self.upsample(x)
        x = self.pretrained_model(x)
        return x


if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    cat_dataset = CatDataset(transform=transform)
    encoder = init_encoder("inceptionv4")
    vmmd = init_vmmd()
    vmmd.fit(cat_dataset, embedding_function=encoder)