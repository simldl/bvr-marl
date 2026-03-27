from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as SpaceDict


def to_f32_obs(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    for k, v in list(obs.items()):
        obs[k] = np.asarray(v, dtype=np.float32)
    return obs


def coerce_obs_to_space(space: SpaceDict, ob: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Ensure `ob` matches `space` exactly: keys, shapes, dtypes, bounds.
    Missing keys are zero-filled; present keys are reshaped/cast; values clipped.
    If reshape fails due to size mismatch, pad/truncate to fit.

    Fast path: if a value is already an ndarray with the correct shape and dtype,
    only clipping is applied (no extra asarray/reshape calls).
    """
    out: dict[str, np.ndarray] = {}
    for k, subspace in space.spaces.items():
        v = ob.get(k, None)
        if isinstance(subspace, Box):
            shape = subspace.shape
            dtype = subspace.dtype or np.float32
            if v is None:
                v = np.zeros(shape, dtype=dtype)
            elif isinstance(v, np.ndarray) and v.shape == shape and v.dtype == dtype:
                # Fast path: correct shape and dtype — only apply bounds clip
                v = np.clip(v, subspace.low, subspace.high)
            else:
                v = np.asarray(v, dtype=dtype)
                # Try to reshape, but handle size mismatches gracefully
                try:
                    v = v.reshape(shape)
                except ValueError as e:
                    if "cannot reshape" in str(e):
                        # Handle size mismatch by padding or truncating
                        expected_size = np.prod(shape)
                        current_size = v.size

                        if current_size < expected_size:
                            # Pad with zeros
                            padding = np.zeros(expected_size - current_size, dtype=dtype)
                            v = np.concatenate([v.flatten(), padding])
                        else:
                            # Truncate to fit
                            v = v.flatten()[:expected_size]

                        v = v.reshape(shape)
                    else:
                        raise
                v = np.clip(v, subspace.low, subspace.high).astype(dtype, copy=False)
        else:
            if v is None:
                # Deterministic zero construction — no stochastic sample()
                shape = getattr(subspace, "shape", (1,))
                v = np.zeros(shape, dtype=np.float32)
            else:
                v = np.asarray(v)
        out[k] = v
    return out


def _finite_bounds_like(arr: np.ndarray, pad: float = 1e6) -> tuple[np.ndarray, np.ndarray]:
    """
    Create wide-but-finite bounds for a numeric vector if no explicit bounds exist.
    """
    # Try to use magnitude-aware bounds if data is non-zero; else default +/- pad
    mag = np.maximum(np.abs(arr), 1.0)
    low = -mag * 100.0
    high = +mag * 100.0
    return low.astype(np.float32), high.astype(np.float32)


def space_from_array(arr: np.ndarray) -> Box:
    arr = np.asarray(arr, dtype=np.float32)
    low, high = _finite_bounds_like(arr)
    return Box(low=low, high=high, dtype=np.float32)


def spaces_from_sample_observation(sample: dict[str, Any]) -> SpaceDict:
    """
    Build a gymnasium dict space from a single sample observation dict.
    Each value becomes a Box; shapes are inferred from the arrays.
    """
    subspaces = {}
    for k, v in sample.items():
        v = np.asarray(v, dtype=np.float32)
        subspaces[k] = space_from_array(v)
    return spaces.Dict(subspaces)


def ensure_action_space(action_def: Any, default_dim: int = 9) -> Box:
    """
    Normalize any action spec into a gym Box. If already a gym Space, return it.
    If a dict/array/shape is provided, convert to Box with wide finite bounds.
    """
    if isinstance(action_def, spaces.Space):
        return action_def

    # If an array-like was provided
    if isinstance(action_def, (list, tuple, np.ndarray)):
        action_def = np.asarray(action_def, dtype=np.float32)
        return space_from_array(action_def)

    # If a dict with 'shape'/'low'/'high' exists
    if isinstance(action_def, dict):
        shape = action_def.get("shape", (default_dim,))
        dtype = np.float32
        if "low" in action_def and "high" in action_def:
            low = np.asarray(action_def["low"], dtype=dtype)
            high = np.asarray(action_def["high"], dtype=dtype)
        else:
            # wide but finite
            low = -np.ones(shape, dtype=dtype) * 1e2
            high = +np.ones(shape, dtype=dtype) * 1e2
        return Box(low=low, high=high, dtype=dtype)

    # Fallback: a generic Box of given dim
    return Box(low=-1e2, high=1e2, shape=(default_dim,), dtype=np.float32)
