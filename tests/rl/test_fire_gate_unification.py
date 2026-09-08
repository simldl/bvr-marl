"""``can_fire`` (observation + fire mask) and ``shot_opportunity`` must be one predicate.

They used to be two hand-rolled conjunctions over overlapping gate sets, and every
disagreement produced a metric that could not be reconciled with the behaviour it
described:

* the observation's ``can_fire`` ignored the post-launch cooldown and the per-target
  saturation cap -- its docstring claimed otherwise -- so the fire-gradient mask
  permitted presses that the launch path then vetoed, and
  ``tactical/trigger_precision_rate`` read ~0.000 while the mask was nominally only
  allowing feasible presses;
* ``shot_opportunity_this_step`` ignored the datalink lock, the gimbal limit and the
  range gate, so once a scenario gained an AWACS it undercounted precisely the
  datalink-cued shots the AWACS was added to create.

These tests pin the unification itself, not the individual gates: if someone
reintroduces a second conjunction, they fail.
"""

from __future__ import annotations

import pytest

from bvr_marl_core.aircraft.systems.fire_feasibility import (
    FireGates,
    evaluate_fire_gates,
    missile_cooldown_ok,
    sync_missile_cooldown,
)

ALL_OPEN = dict(
    inventory_ok=True,
    radar_lock=True,
    datalink_lock=False,
    target_in_fov=True,
    gimbal_ok=True,
    radar_range_ok=True,
    cooldown_ok=True,
    target_not_saturated=True,
    remaining_missiles=4,
)


def test_every_gate_is_load_bearing():
    """No gate may be decorative -- closing any one alone must block the shot."""
    assert FireGates(**ALL_OPEN).can_fire

    for gate in (
        "inventory_ok",
        "radar_lock",
        "target_in_fov",
        "gimbal_ok",
        "radar_range_ok",
        "cooldown_ok",
        "target_not_saturated",
    ):
        closed = dict(ALL_OPEN)
        closed[gate] = False
        if gate == "inventory_ok":
            closed["remaining_missiles"] = 0
        assert not FireGates(**closed).can_fire, f"{gate} does not gate can_fire"


def test_cooldown_and_saturation_actually_gate_can_fire():
    """The two gates the observation path silently omitted."""
    assert not FireGates(**{**ALL_OPEN, "cooldown_ok": False}).can_fire
    assert not FireGates(**{**ALL_OPEN, "target_not_saturated": False}).can_fire


def test_datalink_lock_substitutes_for_own_radar_lock():
    """The launch path accepts an AWACS-cued shot, so feasibility must too."""
    cued = FireGates(**{**ALL_OPEN, "radar_lock": False, "datalink_lock": True})

    assert cued.has_lock
    assert cued.can_fire
    # ...and it may be taken beyond the shooter's own radar range.
    assert cued.launch_range_ok
    assert FireGates(
        **{**ALL_OPEN, "radar_lock": False, "datalink_lock": True, "radar_range_ok": False}
    ).can_fire


def test_lock_rate_stays_own_radar_only():
    """`tactical/lock_rate` reports sensor discipline; an AWACS must not flatter it."""
    cued = FireGates(**{**ALL_OPEN, "radar_lock": False, "datalink_lock": True})

    assert cued.has_lock
    assert not cued.radar_lock


def test_no_target_is_not_a_shot_opportunity():
    assert not evaluate_fire_gates(object(), None).can_fire
    assert evaluate_fire_gates(object(), None).veto_reason == "no_target"


@pytest.mark.parametrize(
    "closed,expected",
    [
        ("inventory_ok", "winchester"),
        ("radar_lock", "no_radar_or_datalink_lock"),
        ("target_in_fov", "not_in_fov"),
        ("gimbal_ok", "outside_gimbal_limits"),
        ("cooldown_ok", "missile_cooldown"),
        ("target_not_saturated", "target_saturated"),
    ],
)
def test_veto_reason_names_the_unmet_gate(closed, expected):
    gates = FireGates(**{**ALL_OPEN, closed: False})

    assert gates.veto_reason == expected


def test_veto_reason_is_none_exactly_when_the_shot_is_available():
    assert FireGates(**ALL_OPEN).veto_reason is None


class _Unit:
    pass


def test_cooldown_mirror_is_what_makes_the_observation_agree_with_the_gate():
    """The timer lives in the action-space state dict; the aircraft cannot see it.

    ``sync_missile_cooldown`` is the bridge. Without it the observation reports "ready"
    for the whole post-launch cooldown -- the window in which a policy that just fired
    is most likely to press again.
    """
    unit = _Unit()
    assert missile_cooldown_ok(unit)  # unmirrored: ready, the pre-existing default

    sync_missile_cooldown(unit, 1.8)
    assert not missile_cooldown_ok(unit)

    sync_missile_cooldown(unit, 0.0)
    assert missile_cooldown_ok(unit)


def test_the_duplicate_predicates_are_gone():
    """Guards the unification: neither call site may re-hand-roll the conjunction."""
    from bvr_marl_core.rl.environment.spaces.action_space import weapon_firing

    assert not hasattr(weapon_firing.WeaponFiringHandler, "_sensor_lock_ok")
    assert not hasattr(weapon_firing.WeaponFiringHandler, "_sensor_fov_ok")

    from bvr_marl_core.aircraft.systems import observation_helper

    assert not hasattr(observation_helper.ObservationHelper, "_legacy_get_fire_feasibility")
