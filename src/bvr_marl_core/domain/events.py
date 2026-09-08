from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class DetectionEvent:
    tick: int
    utc_time: datetime
    observer_id: int
    detected_id: int
    confidence: float = 1.0


@dataclass(slots=True)
class LaunchEvent:
    tick: int
    utc_time: datetime
    shooter_id: int
    target_id: int
    weapon_type: str
    weapon_id: int


@dataclass(slots=True)
class HitEvent:
    tick: int
    utc_time: datetime
    weapon_id: int
    target_id: int
    is_kill: bool


@dataclass(slots=True)
class KillEvent:
    tick: int
    utc_time: datetime
    killer_id: int
    victim_id: int
    weapon_type: str


@dataclass(slots=True)
class LockChangeEvent:
    tick: int
    utc_time: datetime
    observer_id: int
    target_id: int | None  # None = lock lost
    lock_type: str  # "radar", "ir", etc.


@dataclass(slots=True)
class ScenarioStartEvent:
    tick: int
    utc_time: datetime
    scenario_id: str
    random_seed: int | None


@dataclass(slots=True)
class ScenarioEndEvent:
    tick: int
    utc_time: datetime
    reason: str  # "timeout" | "all_destroyed" | "terminated"
