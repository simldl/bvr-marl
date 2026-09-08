"""Helpers for seeker-compatible target tracker snapshots."""

from __future__ import annotations

from typing import Any


def build_tracker_info(target: Any) -> dict[str, list[float]]:
    return {
        "position": [target.position.lat, target.position.lon, target.position.alt],
        "velocity": [
            getattr(target, "velocity_x", 0.0),
            getattr(target, "velocity_y", 0.0),
            getattr(target, "velocity_z", 0.0),
        ],
    }
