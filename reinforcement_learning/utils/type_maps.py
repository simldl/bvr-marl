"""Aircraft and missile type mappings.

Maps string identifiers to actual class implementations for configuration.
"""

from aircrafts.types.f22 import F22
from aircrafts.types.eurofighter import Eurofighter
from aircrafts.types.f35 import F35
from aircrafts.types.debug_plane import DebugPlane

from missiles.fox3.amraam import AIM120_AMRAAM
from missiles.fox3.default_missile import LongRangeMissile


# Map string names to aircraft classes
AIRCRAFT_TYPE_MAP = {
    "F22": F22,
    "Eurofighter": Eurofighter,
    "F35": F35,
    "DebugPlane": DebugPlane,
}

# Map string names to missile classes
MISSILE_TYPE_MAP = {
    "AIM120_AMRAAM": AIM120_AMRAAM,
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
