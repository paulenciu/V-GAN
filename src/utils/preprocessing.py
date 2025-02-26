import numpy as np
import torch
from sklearn.preprocessing import normalize

def normalize_images_col(images):
    return normalize(images, axis=0)

def normalize_images_row(images):
    return normalize(images, axis=1)

def standardize_images(images):
    mean = np.mean(images, axis=0)
    std = np.std(images, axis=0)
    return (images - mean) / std