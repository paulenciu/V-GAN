import torch


def extract_and_flatten_images_dataset_3d(dataset: torch.utils.data.Dataset):
    flattened_images = []
    for image, _ in dataset:
        images_flat = image.view(-1)
        flattened_images.append(images_flat)
    return torch.stack(flattened_images)

def unflatten_images_3d(flattened_dataset: torch.Tensor, channels: int, height: int, width: int):
    return flattened_dataset.view(-1, channels, height, width)
