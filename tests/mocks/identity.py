"""Explicit weapon/unit identity for test doubles.

``unittest.mock.Mock`` fabricates any attribute it is asked for, which is exactly
wrong for identity probes. Production code decides what a weapon *is* by asking
questions like "do you carry a weapon track?" or "what contact were you committed
to?"; against a bare Mock every one of those answers yes with a truthy Mock, so a
legacy stub is silently misread as an operational weapon and a stub aircraft looks
like it is carrying missiles.

Declaring the identity surface up front makes "this weapon has no weapon track" an
assertable fact rather than an accident of which attributes a test happened to set.
Use these helpers on any double that reaches the simulator, weapon system, guidance,
or action-processor paths.
"""

from __future__ import annotations

# Identity a weapon is interrogated about. Defaults describe a plain legacy/oracle
# weapon: no operational weapon track and no launch contact.
WEAPON_IDENTITY_DEFAULTS: dict[str, object] = {
    "weapon_track": None,
    "launch_contact_id": None,
    "launch_sensor_id": None,
    "designated_target_id": None,
    "designated_track_id": None,
    "retarget_policy": "locked_override",
    "seduced_by": None,
    "seduced_position": None,
}

# Identity a unit is interrogated about by the weapon/saturation paths.
UNIT_IDENTITY_DEFAULTS: dict[str, object] = {
    "is_missile": False,
    "is_countermeasure": False,
    "is_non_engageable": False,
    "is_destroyed": False,
    "is_mortally_hit": False,
}


def declare_weapon_identity(weapon, **overrides):
    """Pin a weapon double's identity attributes so none can be fabricated."""
    for name, value in {**WEAPON_IDENTITY_DEFAULTS, **overrides}.items():
        setattr(weapon, name, value)
    return weapon


def declare_unit_identity(unit, *, missiles=None, **overrides):
    """Pin a unit double's identity, including a real (empty) weapon list."""
    for name, value in {**UNIT_IDENTITY_DEFAULTS, **overrides}.items():
        setattr(unit, name, value)
    # A real aircraft always owns a list here; a Mock would hand back something
    # non-iterable and break every caller that walks its weapons.
    unit.missiles = [] if missiles is None else missiles
    return unit
