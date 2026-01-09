"""
Padding utilities for fixed-size observation arrays.
Handles padding/truncation of variable-length lists to fixed sizes.
"""
import numpy as np


def pad_generic(items, max_len, dim):
    """
    Pads/truncates a list of vectors/items to uniform length, returns mask.

    Args:
        items: List of vectors/items
        max_len: Maximum length to pad/truncate to
        dim: Dimension of each item

    Returns:
        (arr, mask): Padded array and validity mask
    """
    arr = list(items[:max_len])
    mask = [1.0] * len(arr)
    while len(arr) < max_len:
        arr.append([0.0]*dim)
        mask.append(0.0)
    return np.array(arr, np.float32).reshape(-1, dim), np.array(mask, np.float32)


def pad_indices(indices, max_len):
    """
    Pads/truncates a list of indices, returns mask.

    Args:
        indices: List of integer indices
        max_len: Maximum length

    Returns:
        (arr, mask): Padded indices and mask
    """
    arr = list(indices[:max_len])
    mask = [1.0] * len(arr)
    while len(arr) < max_len:
        arr.append(0)
        mask.append(0.0)
    return np.array(arr, np.int32), np.array(mask, np.float32)


def pad_cached(arrays, max_len, dim):
    """
    Pad/truncate a list of NumPy arrays (e.g. cached sensor tracks) to max_len x dim.

    Args:
        arrays: List of numpy arrays
        max_len: Maximum length
        dim: Dimension of each array

    Returns:
        (padded_array, mask): Stacked array and mask
    """
    arr = list(arrays[:max_len])
    mask = [1.0] * len(arr)
    while len(arr) < max_len:
        arr.append(np.zeros(dim, dtype=np.float32))
        mask.append(0.0)
    return np.stack(arr, axis=0), np.array(mask, np.float32)
