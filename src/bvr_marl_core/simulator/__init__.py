"""Public simulator API for bvr_marl_core."""

from bvr_marl_core.simulator.core.events import (
    UnitDestroyedEvent,
    UnitRegisteredEvent,
    UnitRemovedEvent,
)
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator
from bvr_marl_core.simulator.utils.angles import (
    deg2rad,
    normalize_angle,
    rad2deg,
    signed_yaw_deg_diff,
)
from bvr_marl_core.simulator.utils.geodesics import (
    geodetic_bearing_deg,
    geodetic_distance_km,
    geodetic_to_enu,
)
from bvr_marl_core.simulator.utils.map_limits import MapLimits

__all__ = [
    "Simulator",
    "MapLimits",
    "normalize_angle",
    "signed_yaw_deg_diff",
    "deg2rad",
    "rad2deg",
    "geodetic_bearing_deg",
    "geodetic_distance_km",
    "geodetic_to_enu",
    "Position",
    "UnitDestroyedEvent",
    "UnitRegisteredEvent",
    "UnitRemovedEvent",
]
