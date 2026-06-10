"""Public aircraft API for bvr_marl_core."""

from .aircraft import Aircraft
from .control.awacs_controller import OrbitConfig, create_orbit_controller
from .core.nez import NoEscapeZoneCalculator
from .core.target_prio import TrackPrioritySystem
from .systems.observation_helper import ObservationHelper
from .types.awacs import AWACS
from .types.debug_plane import DebugPlane
from .types.eurofighter import Eurofighter
from .types.f22 import F22
from .types.f35 import F35
from .types.su57 import Su57

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
