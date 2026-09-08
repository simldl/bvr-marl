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
    n = min(len(items), max_len)
    arr = np.zeros((max_len, dim), dtype=np.float32)
    mask = np.zeros(max_len, dtype=np.float32)
    if n > 0:
        arr[:n] = np.array(items[:n], dtype=np.float32).reshape(n, dim)
        mask[:n] = 1.0
    return arr, mask


def pad_tokens(items, max_len, feature_dim):
    """Pad/truncate a list of feature vectors and fold the validity mask into a
    trailing column, yielding self-contained entity tokens.

    Each output row is ``[feature_0 ... feature_{feature_dim-1}, mask]`` where the
    mask is 1.0 for a real entity and 0.0 for a padded slot. This lets the encoder
    recover a per-entity padding mask (for attention) directly from the token,
    instead of a separate parallel mask array.

    Args:
        items: List of length-``feature_dim`` feature vectors.
        max_len: Number of slots to pad/truncate to.
        feature_dim: Feature count per entity, excluding the mask column.

    Returns:
        Array of shape ``(max_len, feature_dim + 1)``; the last column is the mask.
    """
    n = min(len(items), max_len)
    arr = np.zeros((max_len, feature_dim + 1), dtype=np.float32)
    if n > 0:
        arr[:n, :feature_dim] = np.array(items[:n], dtype=np.float32).reshape(n, feature_dim)
        arr[:n, feature_dim] = 1.0
    return arr


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
