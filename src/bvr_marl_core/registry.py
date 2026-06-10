"""Public unit-type registries for bvr_marl_core.

Consumers (e.g. extension packages) should resolve aircraft and missile type
strings through these registries rather than importing concrete classes from
bvr_marl_core.aircraft or bvr_marl_core.missiles directly.  This keeps extension
packages free of static dependencies on internal implementation modules.

Usage::

    from bvr_marl_core.registry import get_aircraft_class, get_missile_class

    AircraftCls = get_aircraft_class("F22")
    unit = AircraftCls(pos, heading, speed, group, map_limits, min_alt, max_alt)
"""

from bvr_marl_core.aircraft.types.awacs import AWACS
from bvr_marl_core.aircraft.types.debug_plane import DebugPlane
from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.aircraft.types.f22 import F22
from bvr_marl_core.aircraft.types.f35 import F35
from bvr_marl_core.aircraft.types.su57 import Su57
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.missiles.fox3.default_missile import LongRangeMissile
from bvr_marl_core.missiles.fox3.k77m import K77M
from bvr_marl_core.missiles.fox3.meteor import Meteor

# ─── Aircraft registry ────────────────────────────────────────────────────────
# Maps normalised string keys to aircraft classes.  Multiple aliases per class
# are supported so that configs written with different naming conventions all
# resolve correctly.

AIRCRAFT_REGISTRY: dict[str, type] = {
    # DebugPlane
    "debug": DebugPlane,
    "debugplane": DebugPlane,
    "debug-plane": DebugPlane,
    "DebugPlane": DebugPlane,
    # Eurofighter
    "eurofighter": Eurofighter,
    "Eurofighter": Eurofighter,
    # F-22
    "f22": F22,
    "F22": F22,
    "f-22": F22,
    # F-35
    "f35": F35,
    "F35": F35,
    "f-35": F35,
    # Su-57
    "su57": Su57,
    "Su57": Su57,
    "su-57": Su57,
    # AWACS
    "awacs": AWACS,
    "AWACS": AWACS,
    "e3": AWACS,
    "e-3": AWACS,
    "e7": AWACS,
    "e-7": AWACS,
}

# ─── Missile registry ─────────────────────────────────────────────────────────

MISSILE_REGISTRY: dict[str, type] = {
    "AIM120_AMRAAM": AIM120_AMRAAM,
    "aim120_amraam": AIM120_AMRAAM,
    "amraam": AIM120_AMRAAM,
    "Meteor": Meteor,
    "meteor": Meteor,
    "K77M": K77M,
    "k77m": K77M,
    "DefaultMissile": LongRangeMissile,
    "default_missile": LongRangeMissile,
    "LongRangeMissile": LongRangeMissile,
    "long_range_missile": LongRangeMissile,
}


def _normalise(name: str) -> str:
    """Return a lower-case, punctuation-stripped version of *name* for fuzzy lookup."""
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def get_aircraft_class(name: str) -> type:
    """Return the aircraft class for the given type name.

    Tries an exact match first, then a normalised (case- and punctuation-
    insensitive) match.

    Raises:
        KeyError: If the name does not match any registered aircraft type.
    """
    if name in AIRCRAFT_REGISTRY:
        return AIRCRAFT_REGISTRY[name]
    key = _normalise(name)
    for k, cls in AIRCRAFT_REGISTRY.items():
        if _normalise(k) == key:
            return cls
    available = sorted({_normalise(k) for k in AIRCRAFT_REGISTRY})
    raise KeyError(f"Unknown aircraft type '{name}'. Available (normalised): {available}")


def get_missile_class(name: str) -> type:
    """Return the missile class for the given type name.

    Raises:
        KeyError: If the name does not match any registered missile type.
    """
    if name in MISSILE_REGISTRY:
        return MISSILE_REGISTRY[name]
    key = _normalise(name)
    for k, cls in MISSILE_REGISTRY.items():
        if _normalise(k) == key:
            return cls
    available = sorted({_normalise(k) for k in MISSILE_REGISTRY})
    raise KeyError(f"Unknown missile type '{name}'. Available (normalised): {available}")


__all__ = [
    "AIRCRAFT_REGISTRY",
    "MISSILE_REGISTRY",
    "get_aircraft_class",
    "get_missile_class",
]
