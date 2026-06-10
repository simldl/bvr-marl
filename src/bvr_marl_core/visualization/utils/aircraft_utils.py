"""Aircraft type utilities."""

from bvr_marl_core.registry import AIRCRAFT_REGISTRY
from bvr_marl_core.registry import get_aircraft_class as _core_get_aircraft_class


def get_aircraft_class(aircraft_type):
    """
    Map aircraft type string to class.

    Args:
        aircraft_type: String identifier for aircraft type

    Returns:
        Aircraft class (falls back to DebugPlane for unknown types)
    """
    try:
        return _core_get_aircraft_class(aircraft_type)
    except KeyError:
        return AIRCRAFT_REGISTRY["DebugPlane"]
