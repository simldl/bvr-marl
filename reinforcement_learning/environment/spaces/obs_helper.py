"""
Observation Helper - Backward compatibility layer.

This module now imports from the new modular structure.
All functions are re-exported from observation.helpers for backward compatibility.
"""
from .observation.helpers import (
    enu_delta_meters as _enu_delta_meters,
    velocity_components as _velocity_components,
    rel_state,
    rel_position,
    rel_velocity,
    pad_generic,
    pad_indices,
    pad_cached,
    extract_feature_matrix,
    build_warn_sector_features,
    onehot_encode,
    mask_for_units,
)

__all__ = [
    '_enu_delta_meters',
    '_velocity_components',
    'rel_state',
    'rel_position',
    'rel_velocity',
    'pad_generic',
    'pad_indices',
    'pad_cached',
    'extract_feature_matrix',
    'build_warn_sector_features',
    'onehot_encode',
    'mask_for_units',
]
