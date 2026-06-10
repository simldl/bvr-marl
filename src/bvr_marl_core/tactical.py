"""Public tactical API for bvr_marl_core.

Provides stable access to engagement-geometry calculators, target-priority
systems, observation helpers, and AWACS orbit primitives.  Consumers (e.g.
extension packages) should import from this module rather than reaching into
bvr_marl_core.aircraft internals directly.
"""

from bvr_marl_core.aircraft.control.awacs_controller import (
    OrbitConfig,
    compute_trailing_center,
    create_orbit_controller,
)
from bvr_marl_core.aircraft.core.nez import NoEscapeZoneCalculator
from bvr_marl_core.aircraft.core.target_prio import TrackPrioritySystem
from bvr_marl_core.aircraft.systems.observation_helper import ObservationHelper
from bvr_marl_core.missiles.missile_parameters import MissileParameters

__all__ = [
    "NoEscapeZoneCalculator",
    "TrackPrioritySystem",
    "ObservationHelper",
    "OrbitConfig",
    "compute_trailing_center",
    "create_orbit_controller",
    "MissileParameters",
]
