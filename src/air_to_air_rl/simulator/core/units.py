from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional

from air_to_air_rl.simulator.core.helpers import Position
from air_to_air_rl.simulator.utils.angles import normalize_angle, yaw_geo_to_math
from air_to_air_rl.simulator.utils.geodesics import geodetic_direct


@dataclass
class Unit(ABC):
    name: str
    position: Position
    yaw_deg: float  # degrees [0,360)
    speed: float  # m/s

    id: int | None = field(default=None, init=False)
    desired_yaw_deg: float = field(init=False)

    def __post_init__(self):
        self.yaw_deg = normalize_angle(self.yaw_deg)
        self.desired_yaw_deg = self.yaw_deg

    @abstractmethod
    def update(self, tick_secs: float, sim) -> list:
        pass

    def physics_step(self, dt, sim) -> list:
        pass

    def substep_update(self, dt: float, sim) -> list:
        return self.physics_step(dt, sim)

    def to_string(self) -> str:
        return (
            f"{self.name}[{self.id}]: "
            f"pos=({self.position.lat:.4f}, {self.position.lon:.4f}, {self.position.alt:.1f}) "
            f"h={self.yaw_deg:.1f}° s={self.speed:.1f}m/s"
        )


Velocity = namedtuple("Velocity", ["vx", "vy", "vz"])


@dataclass
class FlyingUnit(Unit):
    pitch_deg: float = 0.0  # degrees
    roll_deg: float = 0.0  # degrees

    desired_pitch_deg: float = field(init=False)
    desired_roll_deg: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.desired_pitch_deg = self.pitch_deg
        self.desired_roll_deg = self.roll_deg
        self._velocity_cache = None
        self._vel_key = (None, None, None)  # (speed, yaw_deg, pitch_deg)

    def update(self, tick_secs: float, sim) -> list:
        events = []
        if self.speed > 0.0:
            distance = self.speed * tick_secs
            pitch_rad = math.radians(self.pitch_deg)
            vertical = self.speed * math.sin(pitch_rad) * tick_secs
            lat, lon, alt = geodetic_direct(
                self.position.lat,
                self.position.lon,
                self.position.alt,
                self.yaw_deg,
                distance,
                vertical_distance=vertical,
            )
            self.position.lat, self.position.lon, self.position.alt = lat, lon, alt
        return events

    def physics_step(self, tick_secs: float, sim) -> list:
        events = []
        if self.speed > 0.0:
            distance = self.speed * tick_secs
            pitch_rad = math.radians(self.pitch_deg)
            vertical = self.speed * math.sin(pitch_rad) * tick_secs
            lat, lon, alt = geodetic_direct(
                self.position.lat,
                self.position.lon,
                self.position.alt,
                self.yaw_deg,
                distance,
                vertical_distance=vertical,
            )
            self.position.lat, self.position.lon, self.position.alt = lat, lon, alt
        return events

    @property
    def velocity(self) -> Velocity:
        key = (self.speed, self.yaw_deg, self.pitch_deg)
        if self._vel_key == key and self._velocity_cache is not None:
            return self._velocity_cache
        s = self.speed
        pitch_rad = math.radians(self.pitch_deg)
        hor = s * math.cos(pitch_rad)
        # Convert stored geographic yaw (0°=N, 90°=E) to math yaw (0°=E, 90°=N) for vx, vy
        yaw_math = math.radians(yaw_geo_to_math(self.yaw_deg))
        vx = hor * math.cos(yaw_math)  # East
        vy = hor * math.sin(yaw_math)  # North
        vz = s * math.sin(pitch_rad)
        v = Velocity(vx, vy, vz)
        self._velocity_cache = v
        self._vel_key = key
        return v
