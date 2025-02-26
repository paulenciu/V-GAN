import numpy as np
import torch
from sklearn.preprocessing import normalize

def normalize_images(images, axis=0):
    return normalize(images, axis=axis)

def standardize_images(images):
    mean = np.mean(images, axis=0)
    std = np.std(images, axis=0)
    return (images - mean) / std