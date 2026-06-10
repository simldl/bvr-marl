"""Public simulator API for bvr_marl_core."""

from .core.events import UnitDestroyedEvent, UnitRegisteredEvent, UnitRemovedEvent
from .core.helpers import Position
from .simulator import Simulator
from .utils.angles import (
    deg2rad,
    normalize_angle,
    rad2deg,
    signed_yaw_deg_diff,
)
from .utils.geodesics import geodetic_bearing_deg, geodetic_distance_km
from .utils.map_limits import MapLimits

__all__ = [
    "Simulator",
    "MapLimits",
    "normalize_angle",
    "signed_yaw_deg_diff",
    "deg2rad",
    "rad2deg",
    "geodetic_bearing_deg",
    "geodetic_distance_km",
    "Position",
    "UnitDestroyedEvent",
    "UnitRegisteredEvent",
    "UnitRemovedEvent",
]
