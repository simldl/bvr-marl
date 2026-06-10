import math
from collections import deque

from bvr_marl_core.simulator.utils.geodesics import geodetic_distance_km


class MovingAverageFilter:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def add(self, value: float):
        self.values.append(value)

    def average(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0


class CircularMovingAverageFilter:
    """Moving average for angles in degrees (0..360), robust to wrap-around."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._s = deque(maxlen=window_size)
        self._c = deque(maxlen=window_size)

    def add(self, angle_deg: float):
        a = math.radians(angle_deg % 360.0)
        self._s.append(math.sin(a))
        self._c.append(math.cos(a))

    def average(self) -> float:
        if not self._s:
            return 0.0
        s = sum(self._s)
        c = sum(self._c)
        ang = math.degrees(math.atan2(s, c)) % 360.0
        return ang


def distance_m(pos1, pos2):
    return geodetic_distance_km(pos1.lat, pos1.lon, pos1.alt, pos2.lat, pos2.lon, pos2.alt) * 1000.0
