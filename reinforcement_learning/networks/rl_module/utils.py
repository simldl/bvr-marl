"""Utility functions for observation and action space handling."""

from typing import Any
import torch
from gymnasium.spaces import Space, Dict as GymDict
from gymnasium.spaces.utils import flatdim


def space_flatdim(space: Space) -> int:
    """
    Compute the total flattened dimension of a gymnasium space.

    Args:
        space: Gymnasium space (can be Dict, Box, Discrete, etc.)

    Returns:
        Total dimension when flattened
    """
    if isinstance(space, GymDict):
        return sum(flatdim(s) for s in space.spaces.values())
    return flatdim(space)


def flatten_obs_tensor(obs: Any) -> torch.Tensor:
    """
    Flatten observation dict or tensor to 2D tensor (batch, features).

    Handles both dict observations (concatenates all values) and
    raw tensor observations (ensures 2D shape).

    Args:
        obs: Observation dict or tensor

    Returns:
        Flattened 2D tensor (batch_size, total_features)
    """
    if isinstance(obs, dict):
        parts = []
        for k in sorted(obs.keys()):  # Stable ordering
            v = obs[k]
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            parts.append(v.view(v.shape[0], -1))
        return torch.cat(parts, dim=-1)

    if not isinstance(obs, torch.Tensor):
        obs = torch.as_tensor(obs)
    return obs.view(obs.shape[0], -1)


def get_action_mask_from_obs(obs: Any, key: str = "action_mask") -> torch.Tensor:
    """
    Extract action mask from observation dict if present.

    Args:
        obs: Observation (dict or tensor)
        key: Key to look for in observation dict

    Returns:
        Action mask tensor (float32) or None if not present
    """
    if isinstance(obs, dict) and key in obs:
        mask = obs[key]
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        return mask.to(dtype=torch.float32)
    return None
