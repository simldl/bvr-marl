from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ControlCommand:
    """Generic flight-control command issued to a platform."""

    unit_id: int
    heading_cmd_deg: float | None = None
    speed_cmd_ms: float | None = None
    altitude_cmd_m: float | None = None


@dataclass(slots=True)
class FireCommand:
    """Command to fire a weapon at a target."""

    unit_id: int
    weapon_type: str
    target_id: int


@dataclass(slots=True)
class ManeuverCommand:
    """Named maneuver (e.g. 'crank', 'notch', 'beam')."""

    unit_id: int
    maneuver_name: str
    parameters: dict = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
