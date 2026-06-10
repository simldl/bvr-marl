from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from bvr_marl_core.simulator.utils.geodesics import geodetic_bearing_deg, geodetic_distance_km


@dataclass
class Position:
    lat: float
    lon: float
    alt: float  # meters

    def copy(self) -> Position:
        return Position(self.lat, self.lon, self.alt)


def units_distance_km(a, b) -> float:
    return geodetic_distance_km(
        a.position.lat,
        a.position.lon,
        a.position.alt,
        b.position.lat,
        b.position.lon,
        b.position.alt,
    )


def units_bearing(a, b) -> float:
    return geodetic_bearing_deg(a.position.lat, a.position.lon, b.position.lat, b.position.lon)


T = TypeVar("T", int, float)


def clamp[T: (int, float)](x: T, lo: T, hi: T) -> T:
    if lo > hi:
        lo, hi = hi, lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
