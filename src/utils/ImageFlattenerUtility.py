import torch


def flatten_images_3d(dataset: torch.utils.data.Dataset):
    flattened_images = []
    for image, _ in dataset:
        images_flat = image.view(-1)
        flattened_images.append(images_flat)
    return torch.stack(flattened_images)

def unflatten_images_3d(flattened_dataset: torch.Tensor, channels: int, height: int, width: int):
    reshaped_images = []
    for flattened_image in flattened_dataset:
        image_3d = flattened_image.view(channels, height, width)
        reshaped_images.append(image_3d)
    return torch.stack(reshaped_images)
