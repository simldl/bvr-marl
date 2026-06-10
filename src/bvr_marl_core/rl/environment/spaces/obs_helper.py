"""
Observation Helper - Backward compatibility layer.

This module now imports from the new modular structure.
All functions are re-exported from observation.helpers for backward compatibility.
"""

from .observation.helpers import (
    build_warn_sector_features,
    extract_feature_matrix,
    mask_for_units,
    onehot_encode,
    pad_cached,
    pad_generic,
    pad_indices,
    rel_position,
    rel_state,
    rel_velocity,
)
from .observation.helpers import (
    enu_delta_meters as _enu_delta_meters,
)
from .observation.helpers import (
    velocity_components as _velocity_components,
)

__all__ = [
    "_enu_delta_meters",
    "_velocity_components",
    "rel_state",
    "rel_position",
    "rel_velocity",
    "pad_generic",
    "pad_indices",
    "pad_cached",
    "extract_feature_matrix",
    "build_warn_sector_features",
    "onehot_encode",
    "mask_for_units",
]
