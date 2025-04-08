import torch
import torchvision.transforms
from PIL import Image
from src.data.dataset.MVTecADDataset import MVTecADDataset

from torchvision import transforms
from src.data.dataset_type import DatasetType


def load_data(dataset_type: DatasetType, category, image_size=(128, 128), custom_transform=None, standardize=False, train=True):

    if custom_transform is not None:
        transform = custom_transform
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    x_test = dataset_type.load(category=category, transform=transform, train=train, normalize=standardize)
    y_test = x_test.labels

    if not isinstance(x_test.dataset, tuple):
        #x_test = [transform((Image.open(image_path).convert('RGB'))) for image_path in x_test.dataset]
        x_test = [x_test[i][0] for i in range(len(x_test))]
        x_test = torch.stack(x_test)
    else:
        x_test = torch.stack(x_test.dataset)

    return x_test, y_test
