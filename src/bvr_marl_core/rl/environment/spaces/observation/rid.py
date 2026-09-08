"""Radar Identification (RID) helpers.

Once a contact is held on a weapons-quality (locked / high-confidence) track, the
simulator reveals the enemy's *exact* airframe type and its armament to the agent
observation, emulating the identity a modern fire-control radar / NCTR extracts
from the return signature. Below a weapons-quality track the identity is unknown
and every RID feature is 0.0.

Type identities are exported as small normalized indices into stable, canonical
orderings of the registered aircraft and missile classes (sorted by class name so
the mapping is deterministic across runs). Index 0 is reserved for "unknown", so
a resolved type maps to (rank + 1) / N in (0, 1].
"""

from functools import lru_cache

from bvr_marl_core.registry import AIRCRAFT_REGISTRY, MISSILE_REGISTRY


@lru_cache(maxsize=1)
def _airframe_order() -> dict[str, int]:
    """Class-name -> rank (0-based) over the unique registered aircraft classes."""
    names = sorted({cls.__name__ for cls in AIRCRAFT_REGISTRY.values()})
    return {name: i for i, name in enumerate(names)}


@lru_cache(maxsize=1)
def _weapon_order() -> dict[str, int]:
    """Class-name -> rank (0-based) over the unique registered missile classes."""
    names = sorted({cls.__name__ for cls in MISSILE_REGISTRY.values()})
    return {name: i for i, name in enumerate(names)}


def airframe_id_norm(unit) -> float:
    """Normalized airframe-type id in (0, 1]; 0.0 if the type is unregistered."""
    order = _airframe_order()
    rank = order.get(type(unit).__name__)
    if rank is None:
        return 0.0
    return (rank + 1) / len(order)


def weapon_id_norm(unit) -> float:
    """Normalized primary-weapon id in (0, 1]; 0.0 if unknown.

    Uses the first configured missile type (the airframe's primary BVR weapon).
    """
    missile_types = getattr(getattr(unit, "weapons", None), "missile_types", None)
    if not missile_types:
        return 0.0
    order = _weapon_order()
    rank = order.get(missile_types[0].__name__)
    if rank is None:
        return 0.0
    return (rank + 1) / len(order)


def missiles_remaining_norm(unit) -> float:
    """Enemy missiles remaining, normalized by its own max load; 0.0 if unknown."""
    weapons = getattr(unit, "weapons", None)
    if weapons is None:
        return 0.0
    remaining = getattr(unit, "remaining_missiles", None)
    if remaining is None:
        remaining = getattr(weapons, "remaining_missiles", None)
    max_missiles = getattr(weapons, "max_missiles", None)
    if remaining is None or not max_missiles:
        return 0.0
    return max(0.0, min(1.0, float(remaining) / float(max_missiles)))
