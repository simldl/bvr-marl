"""Pydantic v2 simulation configuration model."""

from __future__ import annotations

from bvr_marl_core.schema.base import (
    VersionedModel,  # SimulationConfig is a top-level document model
)


class SimulationConfig(VersionedModel):
    tick_secs: float = 1.0
    substep_engage_distance_km: float = 2.0
    missile_hit_radius_m: float = 500.0
    gun_lethal_radius_m: float = 5.0
    enable_replay_recording: bool = False
    replay_output_path: str | None = None
