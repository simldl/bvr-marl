"""Public aircraft API for bvr_marl_core."""

from bvr_marl_core.aircraft.aircraft import Aircraft
from bvr_marl_core.aircraft.control.awacs_controller import OrbitConfig, create_orbit_controller
from bvr_marl_core.aircraft.core.nez import NoEscapeZoneCalculator
from bvr_marl_core.aircraft.core.target_prio import TrackPrioritySystem
from bvr_marl_core.aircraft.systems.observation_helper import ObservationHelper
from bvr_marl_core.aircraft.types.awacs import AWACS
from bvr_marl_core.aircraft.types.debug_plane import DebugPlane
from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.aircraft.types.f22 import F22
from bvr_marl_core.aircraft.types.f35 import F35
from bvr_marl_core.aircraft.types.su57 import Su57

__all__ = [
    "Aircraft",
    "AWACS",
    "DebugPlane",
    "Eurofighter",
    "F22",
    "F35",
    "Su57",
    "NoEscapeZoneCalculator",
    "TrackPrioritySystem",
    "ObservationHelper",
    "OrbitConfig",
    "create_orbit_controller",
]
