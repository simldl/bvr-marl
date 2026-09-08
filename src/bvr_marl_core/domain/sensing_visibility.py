"""The single definition of "can this observer sense that unit at all?".

Sensor invisibility is a *support-asset* rule: an AWACS is on station to PROVIDE radar
coverage, not to be found by it. It is deliberately NOT the same rule as
``is_non_engageable``:

* ``is_non_engageable`` is RULES OF ENGAGEMENT -- it refuses the launch. A refused
  launch still costs everything upstream of it: the unit is swept, its RCS and SNR are
  computed, it becomes a detection, then a track, then a contact, and it OCCUPIES A
  TARGET SLOT.
* ``is_sensor_invisible`` is DETECTION -- the observer never enumerates the unit, so
  none of that work happens and no slot is taken.

Both flags are kept because a scenario may legitimately want a visible-but-protected
unit (or, in principle, the reverse), so neither may silently imply the other.

Two properties this module exists to guarantee, because they were previously restated
at four call sites that could drift apart:

1. **It is hostile-only.** A friendly AWACS stays fully visible to the team it is
   supporting -- otherwise the datalink picture it exists to provide would disappear
   along with it.
2. **It is applied at ENUMERATION**, before any per-candidate work. That is what makes
   this a performance property as well as a behavioural one: measured ~20% faster ticks
   on the 4v4 stage (102.4 -> 82.0 ms/tick, radar candidates per call 9.96 -> 8.01).

This is emphatically NOT modelled as ``rcs = 0``. The airframe stays physically large
(``AWACS.Config.rcs`` is 100.0). Zeroing RCS would keep paying the whole per-candidate
sensor chain, would still let the unit take a slot once a false alarm or a datalink
track produced one, and would hide a 150-tonne aircraft from FRIENDLY sensors too.
"""

from __future__ import annotations

from typing import Any

__all__ = ["is_sensor_invisible_to", "sensible_hostiles"]


def is_sensor_invisible_to(unit: Any, observer_group: Any) -> bool:
    """True when ``unit`` must not be enumerated by a sensor owned by ``observer_group``.

    Returns False for own-team units regardless of their flag: invisibility is a rule
    about being seen by the ENEMY, and a team must not be blinded to its own assets.
    A unit that never heard of the flag is visible, so this is safe on any object.
    """
    if getattr(unit, "group", None) == observer_group:
        return False
    return bool(getattr(unit, "is_sensor_invisible", False))


def sensible_hostiles(units: Any, observer_group: Any) -> list[Any]:
    """The subset of ``units`` an observer on ``observer_group`` may try to detect.

    Filters hostility and invisibility only. Callers add their own further rules (self,
    missiles, countermeasures, ROE), because those differ per sensor.
    """
    return [
        unit
        for unit in units
        if getattr(unit, "group", None) != observer_group
        and not is_sensor_invisible_to(unit, observer_group)
    ]
