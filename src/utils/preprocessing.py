import numpy as np
import scipy
import torch
from sklearn.preprocessing import normalize

def normalize_features(images):
    return normalize(images, axis=0)

def normalize_images(images):
    return normalize(images, axis=1)

def min_max_scaling(x):
    max_x = np.max(x)
    min_x = np.min(x)
    if max_x > min_x:
        x = (x - min_x) / (max_x - min_x)
    return x

def normalize_images_col_softmax(images):
    return scipy.special.softmax(images, axis=0)

def standardize_images(images):
    mean = np.mean(images, axis=0)
    std = np.std(images, axis=0)
    return (images - mean) / std