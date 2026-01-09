from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Optional
import math
from simulator.utils.angles import normalize_angle, yaw_geo_to_math
from simulator.utils.geodesics import geodetic_direct
from simulator.core.helpers import Position


@dataclass
class Unit(ABC):
    name: str
    position: Position
    yaw_deg: float          # degrees [0,360)
    speed: float            # m/s

    id: Optional[int] = field(default=None, init=False)
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
        return (f"{self.name}[{self.id}]: "
                f"pos=({self.position.lat:.4f}, {self.position.lon:.4f}, {self.position.alt:.1f}) "
                f"h={self.yaw_deg:.1f}° s={self.speed:.1f}m/s")

from collections import namedtuple
Velocity = namedtuple("Velocity", ["vx", "vy", "vz"])

@dataclass
class FlyingUnit(Unit):
    pitch_deg: float = 0.0      # degrees
    roll_deg: float = 0.0       # degrees

    desired_pitch_deg: float = field(init=False)
    desired_roll_deg: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.desired_pitch_deg = self.pitch_deg
        self.desired_roll_deg = self.roll_deg

    def update(self, tick_secs: float, sim) -> list:
        events = []
        if self.speed > 0.0:
            distance = self.speed * tick_secs
            pitch_rad = math.radians(self.pitch_deg)
            vertical = self.speed * math.sin(pitch_rad) * tick_secs
            lat, lon, alt = geodetic_direct(
                self.position.lat, self.position.lon, self.position.alt,
                self.yaw_deg, distance, vertical_distance=vertical
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
                self.position.lat, self.position.lon, self.position.alt,
                self.yaw_deg, distance, vertical_distance=vertical
            )
            self.position.lat, self.position.lon, self.position.alt = lat, lon, alt
        return events

    @property
    def velocity(self) -> Velocity:
        s = self.speed
        hor = s * math.cos(math.radians(self.pitch_deg))
        # Convert stored geographic yaw (0°=N, 90°=E) to math yaw (0°=E, 90°=N) for vx, vy
        yaw_math = math.radians(yaw_geo_to_math(self.yaw_deg))
        vx = hor * math.cos(yaw_math)   # East
        vy = hor * math.sin(yaw_math)   # North
        vz = s   * math.sin(math.radians(self.pitch_deg))
        return Velocity(vx, vy, vz)