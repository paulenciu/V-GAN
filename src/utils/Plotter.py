import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from torchvision import transforms
import torch_two_sample as tts
import torch.nn.functional as F


from src.utils.BigUBuilder import create_big_u
from src.utils.ImageFlattenerUtility import flatten_images_3d
from src.vmmd.VMMDRotationMapping import VMMDRotationMapping
from src.vmmd.vmmd import VMMD


def plot_masking(big_u, dataset):
    # Prepare a figure to display 10 pairs of images (stacked vertically)
    fig, axes = plt.subplots(12, 5, figsize=(12, 40))  # 10 rows, 2 columns
    fig.suptitle("Original and Manipulated Images", fontsize=16)

    u = big_u.view(3, 32, 32).permute(1, 2, 0).cpu().numpy()
    mask_r = u[..., 0]  # Red channel
    mask_g = u[..., 1]  # Green channel
    mask_b = u[..., 2]  # Blue channel

    axes[0, 1].imshow(mask_r, cmap="Reds")
    axes[0, 1].set_title("Red Channel Mask")

    axes[0, 2].imshow(mask_g, cmap="Greens")
    axes[0, 2].set_title("Green Channel Mask")

    axes[0, 3].imshow(mask_b, cmap="Blues")
    axes[0, 3].set_title("Blue Channel Mask")

    axes[1, 1].imshow(u)
    axes[1, 1].set_title("Original Mask")

    mean_combined_mask = torch.tensor(np.mean(u, axis=-1))
    axes[1, 2].imshow(mean_combined_mask, cmap="gray")
    axes[1, 2].set_title("Combined Mask (mean)")

    max_combined_mask = torch.tensor(np.max(u, axis=-1))
    axes[1, 3].imshow(max_combined_mask, cmap="gray")
    axes[1, 3].set_title("Combined Mask (max)")

    min_combined_mask = torch.tensor(np.min(u, axis=-1))
    axes[1, 4].imshow(min_combined_mask, cmap="gray")
    axes[1, 4].set_title("Combined Mask (min)")

    for i in range(0, 10):  # Process the first 10 images in the dataset
        original_image = dataset[i][0]  # Load the i-th image and send it to the specified device
        flattened_image = original_image.view(-1)
        manipulated_image = big_u * flattened_image

        flattened_mean_combined_mask = mean_combined_mask.view(-1).repeat(3)
        manipulated_image_2 = flattened_mean_combined_mask * flattened_image

        flattened_max_combined_mask = max_combined_mask.view(-1).repeat(3)
        manipulated_image_3 = flattened_max_combined_mask * flattened_image

        flattened_min_combined_mask = min_combined_mask.view(-1).repeat(3)
        manipulated_image_4 = flattened_min_combined_mask * flattened_image

        # Convert tensors to NumPy arrays
        original_image_np = original_image.view(3, 32, 32).permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
        manipulated_image_np = manipulated_image.view(3, 32, 32).permute(1, 2, 0).cpu().numpy()
        manipulated_image_2_np = manipulated_image_2.view(3, 32, 32).permute(1, 2, 0).cpu().numpy()
        manipulated_image_3_np = manipulated_image_3.view(3, 32, 32).permute(1, 2, 0).cpu().numpy()
        manipulated_image_4_np = manipulated_image_4.view(3, 32, 32).permute(1, 2, 0).cpu().numpy()

        # Plot the original image
        axes[i + 2, 0].imshow(original_image_np)
        axes[i + 2, 0].set_title(f"Original {i + 1}")
        axes[i + 2, 0].axis("off")

        # Plot the manipulated image
        axes[i + 2, 1].imshow(manipulated_image_np)
        axes[i + 2, 1].set_title(f"mask per channel")
        axes[i + 2, 1].axis("off")

        # Plot the manipulated image
        axes[i + 2, 2].imshow(manipulated_image_2_np)
        axes[i + 2, 2].set_title(f"mean mask")
        axes[i + 2, 2].axis("off")

        axes[i + 2, 3].imshow(manipulated_image_3_np)
        axes[i + 2, 3].set_title(f"max mask")
        axes[i + 2, 3].axis("off")

        axes[i + 2, 4].imshow(manipulated_image_4_np)
        axes[i + 2, 4].set_title(f"min mask")
        axes[i + 2, 4].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust spacing for the title
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import torch


def tensor_to_image(tensor):
    """Convert a tensor to a numpy image for visualization."""
    tensor = tensor.detach().cpu().numpy()
    if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:  # (C, H, W)
        tensor = np.transpose(tensor, (1, 2, 0))  # Convert to (H, W, C)
    return tensor

def visualize_tensor_processing(batch, n_samples, generator, n_channels, width, height):

    def normalize_to_01(tensor):
        tensor = tensor.to("cpu")
        tensor = tensor - tensor.min()  # Shift to make minimum 0
        tensor = tensor / tensor.max()  # Scale to make maximum 1
        return tensor

    # Step 1: Generate fake_subspaces from the generator
    fake_subspaces = generator.generate_subspaces(n_samples).to(torch.float32)

    # Step 2: Unsqueeze and repeat for broadcasting
    fake_subspaces = fake_subspaces.unsqueeze(1).repeat(1, 3, 1, 1)  # Shape: (n_samples, 3, width, height)

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
        # Row i, Column 1: Used mapping (fake_subspaces)
        axs[i, 0].imshow(tensor_to_image(fake_subspaces[i]))
        axs[i, 0].set_title(f"Mapping {i + 1}")
        axs[i, 0].axis("off")

        # Row i, Column 2: Original image (batch)
        axs[i, 1].imshow(tensor_to_image(batch[i]))
        axs[i, 1].set_title(f"Original {i + 1}")
        axs[i, 1].axis("off")

        # Row i, Column 3: Projection (processed_fake_subspaces)
        axs[i, 2].imshow(tensor_to_image(normalize_to_01(processed_fake_subspaces[i])))
        axs[i, 2].set_title(f"Projection {i + 1}")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def visualise_rotations_of_vmmd(model: VMMDRotationMapping, n_samples=2):

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    # Load CIFAR-10 dataset and filter for cat images (label = 3)
    dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                           transform=transforms.ToTensor())
    cats_dataset = [(img, label) for (img, label) in dataset if label == 3]
    sample_indices = np.arange(min(n_samples, len(cats_dataset)))  # Ensure it doesn't exceed dataset size
    batch = [cats_dataset[i][0] for i in sample_indices]


    batch = torch.stack(batch).to(torch.float32).to(device)
    rotations_batch = model.generate_subspaces(n_samples).to(torch.float32).to(device)

    rotated_images = model.rotate_images(rotations_batch, batch)

    fig, axs = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))

    for i in range(n_samples):
        # Row i, Column 1: Used mapping (fake_subspaces)
        axs[i, 0].imshow(tensor_to_image(rotations_batch[i]))
        axs[i, 0].set_title(f"Mapping {i + 1}")
        axs[i, 0].axis("off")

        # Row i, Column 2: Original image (batch)
        axs[i, 1].imshow(tensor_to_image(batch[i]))
        axs[i, 1].set_title(f"Original {i + 1}")
        axs[i, 1].axis("off")

        # Row i, Column 3: Projection (processed_fake_subspaces)
        axs[i, 2].imshow(tensor_to_image(rotated_images[i]))
        axs[i, 2].set_title(f"Projection {i + 1}")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def visualise_linear_mapping_of_vmmd(model, n_samples=2):

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    # Load CIFAR-10 dataset and filter for cat images (label = 3)
    dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                           transform=transforms.ToTensor())
    cats_dataset = [(img, label) for (img, label) in dataset if label == 3]

    # Define the number of samples and select a subset of cat images
    sample_indices = np.arange(min(n_samples, len(cats_dataset)))  # Ensure it doesn't exceed dataset size
    batch = [cats_dataset[i][0] for i in sample_indices]
    batch = torch.stack(batch).to(torch.float32).to(device)

    # Call the updated visualization function
    visualize_tensor_processing(batch, n_samples, model, 3, 32, 32)

def visualise_single_masking_of_vmmd(model):
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                           transform=transforms.ToTensor())
    cats_dataset = [(img, label) for (img, label) in dataset if label == 3]

    n_samples = 2
    sample_indices = np.arange(n_samples)
    X_sample = torch.utils.data.Subset(cats_dataset, sample_indices)
    X_sample = flatten_images_3d(X_sample, device)

    u = model.generate_subspaces(n_samples).repeat(1, 3)

    uX_data = u * \
              X_sample + \
              torch.mean(X_sample, dim=0) * (~u)

    mmd = tts.MMDStatistic(n_samples, n_samples)
    mmd_val, distances = mmd(X_sample, uX_data, alphas=[0.01], ret_matrix=True)

    big_u, unique_subspaces, proba = create_big_u(u)

    plot_masking(big_u, cats_dataset)

    unique_subspaces = [str(unique_subspaces[i] * 1)
                        for i in range(unique_subspaces.shape[0])]

    print(pd.DataFrame({'subspace': unique_subspaces, 'probability': proba}))
    print(np.sum(proba))


def plot_pvals(vmmd, x, min=100, max=1000, step=100):
    pvals = {}
    for i in range(min, max, step):
        df = vmmd.check_if_myopic(x, count=i)
        pvals[i] = df.iat[0, 0]

    plt.plot(list(pvals.keys()), list(pvals.values()))
    plt.xlabel("Number of samples")
    plt.ylabel("P-value")
    plt.title("Number of samples vs. P-value")
    plt.show()
