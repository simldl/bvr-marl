"""Shared helpers for the effectiveness submodels."""

from __future__ import annotations

from typing import Any


def clamp01(x: float) -> float:
    """Clamp to the valid probability range [0, 1]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def attr_float(obj: Any, name: str, default: float) -> float:
    """Read a float attribute, falling back to ``default`` when it is missing
    or ``None`` (guarded against bad values)."""
    try:
        val = getattr(obj, name, None)
        if val is None:
            return float(default)
        return float(val)
    except (TypeError, ValueError):
        return float(default)
