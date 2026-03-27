"""Aircraft type utilities."""

from air_to_air_rl.aircrafts.types.debug_plane import DebugPlane
from air_to_air_rl.aircrafts.types.eurofighter import Eurofighter
from air_to_air_rl.aircrafts.types.f22 import F22
from air_to_air_rl.aircrafts.types.f35 import F35


def get_aircraft_class(aircraft_type):
    """
    Map aircraft type string to class.

    Args:
        aircraft_type: String identifier for aircraft type

    Returns:
        Aircraft class
    """
    aircraft_map = {"F22": F22, "Eurofighter": Eurofighter, "F35": F35, "DebugPlane": DebugPlane}
    return aircraft_map.get(aircraft_type, DebugPlane)
