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

    if train:
        return dataset_type.load(category=category, transform=transform, train=True, normalize=standardize)

    x_test = dataset_type.load(category=category, transform=transform, train=False, normalize=standardize)
    y_test = x_test.labels
    return x_test, y_test
