"""Rules of engagement must be ONE rule, shared by the gate and the weapon.

They used to be two. `AircraftWeaponSystem.fire_missile_at_contact` refused a protected
contact with `contact_non_engageable_roe`, while `evaluate_fire_gates` did not model the
rule at all -- so `can_fire`, and with it the observation's can-fire bit, the
shot-opportunity counter and the fire-gradient mask, all advertised a shot the weapon then
silently refused.

Measured on 1v1 self-play with the trigger held down: **92 of 92** gate-passing presses were
rejected as `contact_non_engageable_roe`, for **zero** launches. After the fix the same
probe records zero ROE refusals and the agent shoots its loadout.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bvr_marl_core.aircraft.systems.fire_feasibility import FireGates, roe_engageable


def test_raw_unit_flagged_non_engageable_is_refused():
    assert roe_engageable(SimpleNamespace(id="awacs", is_non_engageable=True)) is False
    assert roe_engageable(SimpleNamespace(id="bandit", is_non_engageable=False)) is True


def test_none_target_is_not_engageable():
    assert roe_engageable(None) is False


def _sim(truth_id, unit):
    return SimpleNamespace(
        evaluator_truth_id_for_contact=lambda shooter, track: truth_id,
        active_units={truth_id: unit} if truth_id is not None else {},
    )


def test_contact_is_resolved_through_the_evaluator_boundary():
    """A TacticalContact carries only a track id, so the rule needs the truth link."""
    contact = SimpleNamespace(track_id="t1")
    shooter = SimpleNamespace(id="A0")

    protected = _sim("u9", SimpleNamespace(is_non_engageable=True))
    assert roe_engageable(contact, aircraft=shooter, simulator=protected) is False

    ordinary = _sim("u9", SimpleNamespace(is_non_engageable=False))
    assert roe_engageable(contact, aircraft=shooter, simulator=ordinary) is True


@pytest.mark.parametrize(
    "sim",
    [
        None,  # no simulator at all
        SimpleNamespace(),  # simulator without the resolver
        SimpleNamespace(evaluator_truth_id_for_contact=lambda s, t: None),  # unresolvable
    ],
)
def test_unevaluable_rule_is_permissive(sim):
    """Closing the gate when the rule cannot be evaluated would forbid EVERY shot.

    The failure has to fall open, not shut: an unresolvable contact is the normal case for
    test doubles and for callers that hold no simulator.
    """
    assert roe_engageable(SimpleNamespace(track_id="t1"), aircraft=None, simulator=sim) is True


def test_roe_is_a_conjunct_of_can_fire():
    """Every other gate passing is not enough if the scenario forbids the target."""
    passing = dict(
        inventory_ok=True,
        radar_lock=True,
        target_in_fov=True,
        gimbal_ok=True,
        radar_range_ok=True,
        weapon_range_ok=True,
        cooldown_ok=True,
        target_not_saturated=True,
        remaining_missiles=4,
    )
    assert FireGates(**passing, roe_ok=True).can_fire is True
    assert FireGates(**passing, roe_ok=False).can_fire is False


def test_roe_defaults_permissive_on_a_bare_firegates():
    """A FireGates built without the rule must behave as it did before it existed."""
    assert FireGates().roe_ok is True


# --- the selectable list: NOT done, and why ------------------------------------------
#
# Keeping protected contacts out of the SELECTABLE list as well would need the ROE flag to
# ride on the track/contact, stamped where truth is already legitimately consulted (the
# detection/tracker boundary). It cannot be done by resolving truth inside
# `select_contact`: `tests/.../test_firewall_target_selection.py` locks the invariant that
# sensor-limited target selection "takes no simulator argument and therefore structurally
# cannot read truth", and that guarantee is worth more than the convenience.
#
# With the gate fix alone the blocker is gone -- `can_fire` is False on a protected contact,
# so it is not counted as a shot opportunity and the fire-gradient mask keeps the trigger
# pinned there. What remains is that the agent can still DESIGNATE one and spend lock and
# attention on it.
