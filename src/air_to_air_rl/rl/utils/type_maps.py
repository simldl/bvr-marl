"""Aircraft and missile type mappings.

Maps string identifiers to actual class implementations for configuration.
"""

from air_to_air_rl.aircrafts.types.awacs import AWACS
from air_to_air_rl.aircrafts.types.debug_plane import DebugPlane
from air_to_air_rl.aircrafts.types.eurofighter import Eurofighter
from air_to_air_rl.aircrafts.types.f22 import F22
from air_to_air_rl.aircrafts.types.f35 import F35
from air_to_air_rl.aircrafts.types.su57 import Su57
from air_to_air_rl.missiles.fox3.amraam import AIM120_AMRAAM
from air_to_air_rl.missiles.fox3.default_missile import LongRangeMissile
from air_to_air_rl.missiles.fox3.k77m import K77M
from air_to_air_rl.missiles.fox3.meteor import Meteor

# Map string names to aircraft classes
AIRCRAFT_TYPE_MAP = {
    "F22": F22,
    "Eurofighter": Eurofighter,
    "F35": F35,
    "Su57": Su57,
    "AWACS": AWACS,
    "DebugPlane": DebugPlane,
}

# Map string names to missile classes
MISSILE_TYPE_MAP = {
    "AIM120_AMRAAM": AIM120_AMRAAM,
    "Meteor": Meteor,
    "K77M": K77M,
    "DefaultMissile": LongRangeMissile,
}


def resolve_aircraft_config(cfg_env):
    """
    Convert string aircraft types to classes and merge with config.

    Args:
        cfg_env: Environment configuration dictionary

    Returns:
        Dictionary with resolved aircraft_types mapping

    Raises:
        ValueError: If aircraft type string is not recognized
    """
    aircraft_config = {}

    if "aircraft_config" in cfg_env:
        ac = cfg_env["aircraft_config"]
        agent_type_str = ac.get("agent_type", "F22")
        opponent_type_str = ac.get("opponent_type", "F22")

        if agent_type_str not in AIRCRAFT_TYPE_MAP:
            raise ValueError(
                f"Unknown aircraft type: {agent_type_str}. "
                f"Available: {list(AIRCRAFT_TYPE_MAP.keys())}"
            )
        if opponent_type_str not in AIRCRAFT_TYPE_MAP:
            raise ValueError(
                f"Unknown aircraft type: {opponent_type_str}. "
                f"Available: {list(AIRCRAFT_TYPE_MAP.keys())}"
            )

        aircraft_config["aircraft_types"] = {
            "agent": AIRCRAFT_TYPE_MAP[agent_type_str],
            "opponent": AIRCRAFT_TYPE_MAP[opponent_type_str],
        }

    return aircraft_config
