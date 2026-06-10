"""Aircraft and missile type mappings.

Maps string identifiers to actual class implementations for configuration.
Uses the core registries — extension code must not import concrete
aircraft/missile classes directly.
"""

from bvr_marl_core.registry import AIRCRAFT_REGISTRY, MISSILE_REGISTRY, get_aircraft_class

# Re-export the canonical maps so existing callers continue to work.
AIRCRAFT_TYPE_MAP = {
    "F22": AIRCRAFT_REGISTRY["F22"],
    "Eurofighter": AIRCRAFT_REGISTRY["Eurofighter"],
    "F35": AIRCRAFT_REGISTRY["F35"],
    "Su57": AIRCRAFT_REGISTRY["Su57"],
    "AWACS": AIRCRAFT_REGISTRY["AWACS"],
    "DebugPlane": AIRCRAFT_REGISTRY["DebugPlane"],
}

MISSILE_TYPE_MAP = {
    "AIM120_AMRAAM": MISSILE_REGISTRY["AIM120_AMRAAM"],
    "Meteor": MISSILE_REGISTRY["Meteor"],
    "K77M": MISSILE_REGISTRY["K77M"],
    "DefaultMissile": MISSILE_REGISTRY["DefaultMissile"],
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
