"""Helper utilities for observation building."""

from bvr_marl_core.rl.environment.spaces.observation.helpers.coordinate_transforms import (
    enu_delta_meters,
    rel_position,
    rel_state,
    rel_velocity,
    velocity_components,
)
from bvr_marl_core.rl.environment.spaces.observation.helpers.feature_extraction import (
    build_warn_sector_features,
    extract_feature_matrix,
    mask_for_units,
    onehot_encode,
)
from bvr_marl_core.rl.environment.spaces.observation.helpers.normalization import (
    OBS_ALT_REF_M,
    OBS_POS_REF_M,
    OBS_RANGE_REF_M,
    OBS_VEL_REF_MPS,
    normalize_pos_vel,
    normalize_range_m,
    to_body_frame,
)
from bvr_marl_core.rl.environment.spaces.observation.helpers.padding_utils import (
    pad_cached,
    pad_generic,
    pad_indices,
    pad_tokens,
)

__all__ = [
    "enu_delta_meters",
    "velocity_components",
    "rel_state",
    "rel_position",
    "rel_velocity",
    "normalize_pos_vel",
    "to_body_frame",
    "OBS_POS_REF_M",
    "OBS_ALT_REF_M",
    "OBS_VEL_REF_MPS",
    "OBS_RANGE_REF_M",
    "normalize_range_m",
    "pad_generic",
    "pad_indices",
    "pad_tokens",
    "pad_cached",
    "extract_feature_matrix",
    "build_warn_sector_features",
    "onehot_encode",
    "mask_for_units",
]
