from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class PlatformState:
    unit_id: int
    team_id: str
    affiliation: str  # "friendly" | "hostile" | "neutral"
    aircraft_type: str
    position_enu: list[float]  # [East, North, Up] metres
    velocity_enu: list[float]  # m/s
    heading_deg: float
    speed_ms: float
    altitude_m: float
    is_alive: bool = True


@dataclass(slots=True)
class SensorState:
    unit_id: int
    sensor_type: str
    is_active: bool
    lock_target_id: int | None = None
    track_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class WeaponState:
    unit_id: int  # owning platform
    weapon_id: int  # unique id of this weapon instance
    weapon_type: str
    is_in_flight: bool
    target_id: int | None
    position_enu: list[float]
    speed_ms: float


@dataclass(slots=True)
class TrackState:
    track_id: int
    target_unit_id: int
    observer_unit_id: int
    position_enu: list[float]
    velocity_enu: list[float]
    confidence: float = 1.0


@dataclass(slots=True)
class EngagementState:
    engagement_id: int
    shooter_id: int
    target_id: int
    weapon_id: int
    launched_at_s: float
    is_active: bool = True


@dataclass(slots=True)
class SimMetadata:
    tick: int
    utc_time: datetime
    elapsed_s: float
    random_seed: int | None
    schema_version: str = "1.0"
