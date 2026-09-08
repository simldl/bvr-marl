"""Aircraft and missile type mappings.

Maps string identifiers to actual class implementations for configuration.
Uses the core registries — extension code must not import concrete
aircraft/missile classes directly.
"""

from bvr_marl_core.registry import AIRCRAFT_REGISTRY, MISSILE_REGISTRY

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

    def _resolve(spec):
        """Resolve a single type string or a list of them to class(es).

        A plain string maps to one class (the whole team flies it); a list maps
        to a list of classes assigned per slot and cycled downstream — this is
        how a heterogeneous formation (e.g. an F-22 leading F-35s) is expressed.
        """
        if isinstance(spec, (list, tuple)):
            return [_resolve(item) for item in spec]
        if spec not in AIRCRAFT_TYPE_MAP:
            raise ValueError(
                f"Unknown aircraft type: {spec}. Available: {list(AIRCRAFT_TYPE_MAP.keys())}"
            )
        return AIRCRAFT_TYPE_MAP[spec]

    if "aircraft_config" in cfg_env:
        ac = cfg_env["aircraft_config"]
        aircraft_config["aircraft_types"] = {
            "agent": _resolve(ac.get("agent_type", "F22")),
            "opponent": _resolve(ac.get("opponent_type", "F22")),
        }

    return aircraft_config
