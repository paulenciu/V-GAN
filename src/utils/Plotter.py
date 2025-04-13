import operator
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from torchvision import transforms
import torch_two_sample as tts
import torch.nn.functional as F

from src.utils.BigUBuilder import calculate_average_u
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d


def tensor_to_image_before(tensor):
    """Convert a tensor to a numpy image for visualization."""
    tensor = tensor.detach().cpu().numpy()
    if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:  # (C, H, W)
        tensor = np.transpose(tensor, (1, 2, 0))  # Convert to (H, W, C)
        tensor = (tensor * 255).astype(np.uint8)


    return tensor

def tensor_to_image(tensor):
    """
    Converts a PyTorch tensor to a numpy image array.
    Ensures that the dtype is compatible with matplotlib.
    """
    if tensor.is_floating_point():
        tensor = tensor.to(torch.float32)  # Ensure float32 type
    array = tensor.cpu().numpy()  # Convert to numpy
    array = np.clip(array, 0, 1)  # Ensure values are in [0, 1] range
    return array.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

def visualize_tensor_processing(batch, n_samples, generator, n_channels, width, height, unsqueeze_fake_subspace=True):

    def normalize_to_01(tensor):
        tensor = tensor.to("cpu")
        tensor = tensor - tensor.min()  # Shift to make minimum 0
        tensor = tensor / tensor.max()  # Scale to make maximum 1
        return tensor

    # Step 1: Generate fake_subspaces from the generator
    fake_subspaces = generator.generate_subspaces(n_samples).to(torch.float32)

    # Step 2: Unsqueeze and repeat for broadcasting
    if unsqueeze_fake_subspace:
        fake_subspaces = fake_subspaces.unsqueeze(1).repeat(1, 3, 1, 1)  # Shape: (n_samples, 3, width, height)
    else:
        fake_subspaces = fake_subspaces.repeat(1, 3, 1, 1)

    # Step 3: Reshape for matrix multiplication
    fake_subspaces_flat = fake_subspaces.view(-1, width, height)
    batch_flat = batch.view(-1, width, height)

    # Step 4: Perform batch matrix multiplication
    processed_fake_subspaces = torch.bmm(fake_subspaces_flat, batch_flat)  # Shape: (n_samples, width, height)

    # Step 5: Reshape processed tensors back to images
    processed_fake_subspaces = processed_fake_subspaces.view(-1, n_channels, width, height)

    # Visualization: Create one row per sample
    fig, axs = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))

    for i in range(n_samples):
        # Row i, Column 1: Used linear_mapping (fake_subspaces)
        axs[i, 0].imshow(tensor_to_image_before(fake_subspaces[i]))
        axs[i, 0].set_title(f"Mapping {i + 1}")
        axs[i, 0].axis("off")

        # Row i, Column 2: Original image (batch)
        axs[i, 1].imshow(tensor_to_image_before(batch[i]))
        axs[i, 1].set_title(f"Original {i + 1}")
        axs[i, 1].axis("off")

        # Row i, Column 3: Projection (processed_fake_subspaces)
        axs[i, 2].imshow(tensor_to_image_before(normalize_to_01(processed_fake_subspaces[i])))
        axs[i, 2].set_title(f"Projection {i + 1}")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def calculate_angles(rotation_matrices):
    # Extract the relevant elements from the batch of rotation matrices
    R11 = rotation_matrices[:, 0, 0]  # (n,) batch of R[0, 0]
    R21 = rotation_matrices[:, 1, 0]  # (n,) batch of R[1, 0]

    # Calculate the angles using the atan2 function
    angles = torch.atan2(R21, R11)

    return angles

def visualise_rotations_of_vmmd(model, n_samples=10, n_masks=10, path_to_experiment=None, filename="image"):
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    dataset = torchvision.datasets.CIFAR10(root='../cifar10', train=True, download=True,
                                           transform=transforms.ToTensor())
    cats_dataset = [(img, label) for (img, label) in dataset if label == 3]

    sample_indices = np.arange(n_samples)
    X_sample = torch.utils.data.Subset(cats_dataset, sample_indices)

    fig, axis = plt.subplots(n_samples, 1 + n_masks, figsize=(5 * (1 + n_masks), 5 * (n_samples)))

    u = model.generate_subspaces(n_masks).to(device)
    big_u = calculate_average_u(u.detach())
    big_u = big_u.to(torch.float32).to(device)

    for i in range(n_samples):

        image, _ = X_sample[i - 1]
        image = image.to(torch.float32).to(device)

        axis[i, 0].imshow(tensor_to_image(image))
        axis[i, 0].set_title(f"Original {i + 1}")
        axis[i, 0].axis("off")

        u = model.generate_subspaces(n_masks).to(device)
        image = image.unsqueeze(0).repeat(n_masks, 1, 1, 1)
        ux_data = model.rotate_images(u, image)

        big_u = calculate_average_u(u.detach())
        big_u = big_u.to(torch.float32).to(device)

        for j in range(n_masks):
            axis[i, j + 1].imshow(tensor_to_image(ux_data[j]))
            axis[i, j + 1].set_title(f"Projection {j + 1}")
            axis[i, j + 1].axis("off")

    if path_to_experiment is not None:
        fig.tight_layout()
        store_image(path_to_experiment=path_to_experiment, fig=fig, filename=filename)

    plt.tight_layout()
    plt.show()


def visualise_linear_mapping_of_vmmd(model, n_samples=10, unsqueeze_fake_subspace=True):

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    # Load CIFAR-10 data and filter for cat images (label = 3)
    dataset = torchvision.datasets.CIFAR10(root='../cifar10', train=True, download=True,
                                           transform=transforms.ToTensor())
    cats_dataset = [(img, label) for (img, label) in dataset if label == 3]

    # Define the number of samples and select a subset of cat images
    sample_indices = np.arange(min(n_samples, len(cats_dataset)))  # Ensure it doesn't exceed data size
    batch = [cats_dataset[i][0] for i in sample_indices]
    batch = torch.stack(batch).to(torch.float32).to(device)

    # Call the updated visualization function
    visualize_tensor_processing(batch, n_samples, model, 3, 32, 32, unsqueeze_fake_subspace=unsqueeze_fake_subspace)

def visualize_masking_of_vmmd(model, n_samples=10, single=True, n_masks=3):
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    dataset = torchvision.datasets.CIFAR10(root='../cifar10', train=True, download=True,
                                           transform=transforms.ToTensor())
    cats_dataset = [(img, label) for (img, label) in dataset if label == 3]

    sample_indices = np.arange(n_samples)
    X_sample = torch.utils.data.Subset(cats_dataset, sample_indices)

    fig, axis = plt.subplots(n_samples, 2 + n_masks, figsize=(5 * n_samples, 5 * (n_masks + 2)))
    fig.tight_layout()


def store_image(path_to_experiment, fig, filename):
    path_to_directory = Path(path_to_experiment)
    if operator.not_(Path(path_to_directory / 'plots').exists()):
        os.mkdir(path_to_directory / 'plots')

    fig.savefig(path_to_directory / 'plots' / f'{filename}.png')

def visualise_conv_masking_of_vmmd(model, n_samples=10, method="conv_linear", n_masks=3, path_to_experiment=None, filename="image"):

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    dataset = torchvision.datasets.CIFAR10(root='../cifar10', train=True, download=True,
                                           transform=transforms.ToTensor())
    cats_dataset = [(img, label) for (img, label) in dataset if label == 3]

    sample_indices = np.arange(n_samples)
    X_sample = torch.utils.data.Subset(cats_dataset, sample_indices)

    fig, axis = plt.subplots(n_samples + 1, 2 + n_masks, figsize=(5 * (2 + n_masks), 5 * (n_samples + 1)))

    if method == "conv_linear":
        u = model.generate_subspaces(n_masks).repeat(1, 3, 1, 1).to(device)
    elif method == "single":
        u = model.generate_subspaces(n_masks).repeat(1, 3).to(device)
    elif method == "flatten":
        u = model.generate_subspaces(n_masks).to(device)
    else:
        raise NotImplementedError

    big_u = calculate_average_u(u.detach())
    big_u = big_u.to(torch.float32).to(device)

    axis[0, 0].imshow(tensor_to_image(torch.ones(3, 32, 32)))
    axis[0, 0].axis("off")

    for i in range(n_masks):

        if method == "single":
            axis[0, i + 1].imshow(tensor_to_image(u[i].squeeze().view(3, 32, 32)))
        elif method == "conv_linear":
            axis[0, i + 1].imshow(tensor_to_image(u[i]))
        elif method == "flatten":
            axis[0, i + 1].imshow(tensor_to_image(u[i].view(3, 32, 32)))
        else:
            raise NotImplementedError

        axis[0, i + 1].axis("off")
        axis[0, i + 1].set_title(f"Mask {i + 1}")

    if method == "single":
        axis[0, n_masks + 1].imshow(tensor_to_image(big_u.view(3, 32, 32)))
    elif method == "conv_linear":
        axis[0, n_masks + 1].imshow(tensor_to_image(big_u))
    elif method == "flatten":
        axis[0, n_masks + 1].imshow(tensor_to_image(big_u.view(3, 32, 32)))
    else:
        raise NotImplementedError

    axis[0, n_masks + 1].axis("off")
    axis[0, n_masks + 1].set_title(f"Big U Mask")


    for i in range(1, n_samples + 1):

        image, _ = X_sample[i - 1]
        image = image.to(torch.float32).to(device)

        axis[i, 0].imshow(tensor_to_image(image))
        axis[i, 0].set_title(f"Original {i + 1}")
        axis[i, 0].axis("off")

        if method == "conv_linear":
            u = model.generate_subspaces(n_masks).repeat(1, 3, 1, 1).to(device)
            image = image.unsqueeze(0).repeat(n_masks, 1, 1, 1).to(device)
            ux_data = u * image
        elif method == "single":
            u = model.generate_subspaces(n_masks).repeat(1, 3).view(n_masks, -1).to(device)
            image = image.view(-1).unsqueeze(0).repeat(n_masks, 1).to(device)
            ux_data = u * image
        elif method == "flatten":
            u = model.generate_subspaces(n_masks).view(n_masks, -1).to(device)
            image = image.view(-1).unsqueeze(0).repeat(n_masks, 1).to(device)
            ux_data = u * image
        else:
            raise NotImplementedError

        big_u = calculate_average_u(u.detach())
        big_u = big_u.to(torch.float32).to(device)

        for j in range(n_masks):

            if method == "conv_linear":
                axis[i, j + 1].imshow(tensor_to_image(ux_data[j]))
            elif method == "single":
                axis[i, j + 1].imshow(tensor_to_image(ux_data[j].view(3, 32, 32)))
            elif method == "flatten":
                axis[i, j + 1].imshow(tensor_to_image(ux_data[j].view(3, 32, 32)))

            axis[i, j + 1].set_title(f"Projection {j + 1}")
            axis[i, j + 1].axis("off")

        big_u_image = big_u * image[0].squeeze()
        big_u_image = big_u_image.to(torch.float32).to(device)
        if method == "conv_linear":
            axis[i, n_masks + 1].imshow(tensor_to_image(big_u_image))
        elif method == "single" or method == "flatten":
            axis[i, n_masks + 1].imshow(tensor_to_image(big_u_image.view(3, 32, 32)))

        axis[i, n_masks + 1].axis("off")
        axis[i, n_masks + 1].set_title(f"Big U Projection")

    plt.tight_layout()
    plt.show()

    if path_to_experiment is not None:
        fig.tight_layout()
        store_image(path_to_experiment=path_to_experiment, fig=fig, filename=filename)

def get_generator_files(base_dir):
    """
    Recursively iterates through a directory to find all .pt files within subdirectories.

    Parameters:
        base_dir (str): The path to the base 'experiments' folder.

    Returns:
        List[str]: A list of paths to the .pt files.
    """
    generator_files = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.pt'):
                # Append the full path to the generator files list
                generator_files.append(os.path.join(root, file))

    return generator_files

def get_vmmd(file):
    if "single" in file:
        return load_vmmd(file, name="single")
    elif "unscaled" in file:
        return load_vmmd(file, name="flatten")
    elif "conv" in file:

        if "-06" in file:
            return load_vmmd(file, name="conv_linear", generator=GeneratorConvLinearMappingBigSigm())

        return load_vmmd(file, name="conv_linear")
    elif "rotation" in file:
        return load_vmmd(file, name="rotation")
    elif "linear" in file:
        return  load_vmmd(file, name="linear")

    raise NotImplementedError

def plot_pvals_dir(base_dir, x, emb_func, count=1000):
    pt_files = get_generator_files(base_dir)

    for file in pt_files:
        vmmd = get_vmmd(file)
        path_to_experiment = file.split("/models")[0]
        plot_pvals(vmmd, x, emb_func, count, path_to_experiment=path_to_experiment)


def plot_pvals(vmmd, x, emb_func, count=1000, path_to_experiment=None):
    df = vmmd.check_if_myopic(x, count=count, emb_func=emb_func)
    pval_recommended_bw = df.iat[0, 1]

    path_to_experiment = Path(path_to_experiment)
    path = path_to_experiment / 'train_history' / 'generator_loss_0.csv'
    df_g_loss = pd.read_csv(path)
    generator_y = df_g_loss.iloc[:, 0].to_list()

    plt.style.use('ggplot')
    x = np.linspace(1, len(df_g_loss), len(df_g_loss))
    fig, ax = plt.subplots()



    ax.plot(x, generator_y, color="cornflowerblue",
            label="Generator loss", linewidth=2)

    ax.plot([],[], ' ', label= "pval: " + str(pval_recommended_bw))

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    ax.legend(loc="upper right")
    plt.savefig(path_to_experiment / "train_history_w_pval.pdf",
                format="pdf", dpi=1200)