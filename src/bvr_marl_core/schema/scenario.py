"""Pydantic v2 scenario configuration models."""

from __future__ import annotations

from pydantic import Field

from bvr_marl_core.schema.base import NestedModel, VersionedModel


class SensorConfig(NestedModel):
    type: str
    range_km: float
    fov_deg: float
    noise_sigma: float = 0.0


class WeaponConfig(NestedModel):
    type: str
    count: int = 0
    max_range_km: float
    guidance: str = "fox3"


class AgentConfig(NestedModel):
    agent_id: str
    aircraft_type: str
    controller_type: str = "scripted"
    sensors: list[SensorConfig] = Field(default_factory=list)
    weapons: list[WeaponConfig] = Field(default_factory=list)
    initial_position_enu: list[float] = Field(default_factory=lambda: [0.0, 0.0, 8000.0])
    initial_heading_deg: float = 0.0
    initial_speed_ms: float = 250.0


class TeamConfig(NestedModel):
    team_id: str
    affiliation: str
    agents: list[AgentConfig] = Field(default_factory=list)


class ScenarioConfig(VersionedModel):
    scenario_id: str
    description: str = ""
    teams: list[TeamConfig] = Field(default_factory=list)
    max_duration_s: float = 300.0
    random_seed: int | None = None
    map_center_lat: float = 55.0
    map_center_lon: float = 15.0
