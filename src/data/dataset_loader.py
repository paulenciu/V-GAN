from torchvision import transforms

from src.data.dataset_type import DatasetType


def load_data(dataset_type, category, image_size=(128, 128), custom_transform=None):

    if custom_transform is not None:
        transform = custom_transform
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    X_train = dataset_type.load(category=category, transform=transform, train=True)
    X_test = dataset_type.load(category=category, transform=transform, train=False)
    Y_test = X_test.labels
    return X_train, X_test, Y_test
