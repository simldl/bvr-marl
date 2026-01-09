"""
Feature extraction utilities.
Helper functions for building feature vectors and masks.
"""
import numpy as np


def extract_feature_matrix(unit_list, feature_fn, max_len, dim):
    """
    Extracts a feature matrix for arbitrary unit list and returns mask.

    Args:
        unit_list: List of units
        feature_fn: Function that extracts features from a unit
        max_len: Maximum number of units
        dim: Feature dimension

    Returns:
        (features, mask): Feature matrix and validity mask
    """
    feats = []
    for u in unit_list[:max_len]:
        feat = feature_fn(u)
        feats.append(feat if isinstance(feat, (np.ndarray, list, tuple)) else [feat])
    mask = [1.0]*len(feats)
    while len(feats) < max_len:
        feats.append([0.0]*dim)
        mask.append(0.0)
    return np.array(feats, np.float32).reshape(-1, dim), np.array(mask, np.float32)


def build_warn_sector_features(unit, warn_dict, sectors=8):
    """
    Builds a sector warning feature (missile warnings per sector).

    Args:
        unit: Unit for reference (unused currently)
        warn_dict: Dict with sector index (int) -> number of active warnings
        sectors: Number of warning sectors

    Returns:
        np.ndarray: Warning counts per sector
    """
    arr = np.zeros(sectors, dtype=np.float32)
    for idx, count in warn_dict.items():
        if 0 <= idx < sectors:
            arr[idx] = count
    return arr


def onehot_encode(idx, size):
    """
    One-hot encoding for an index.

    Args:
        idx: Index to encode
        size: Size of one-hot vector

    Returns:
        np.ndarray: One-hot encoded vector
    """
    arr = np.zeros(size, dtype=np.float32)
    if 0 <= idx < size:
        arr[idx] = 1.0
    return arr


def mask_for_units(unit_list, max_len):
    """
    Create mask for existing/active units.

    Args:
        unit_list: List of units (may contain None)
        max_len: Maximum length

    Returns:
        np.ndarray: Mask (1.0 for valid units, 0.0 otherwise)
    """
    mask = [1.0 if u is not None else 0.0 for u in unit_list[:max_len]]
    while len(mask) < max_len:
        mask.append(0.0)
    return np.array(mask, np.float32)
