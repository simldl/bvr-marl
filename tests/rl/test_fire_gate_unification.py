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
    sync_designated_contact,
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


class _Contact:
    """Stands in for a TacticalContact; only identity and the flags below are read."""

    def __init__(self, name, engageable=True):
        self.name = name
        self.engageable = engageable
        self.suspect_deception = False
        self.is_missile = False

    def __repr__(self):
        return f"<{self.name}>"


def test_designated_contact_mirror_feeds_the_observation_cue_target():
    """The cues must describe the DESIGNATED contact, not whatever the radar holds.

    Unifying the GATES was not enough: the two call sites still resolved different
    TARGETS. The launch path and `shot_opportunities` ask about `selected_target`;
    the observation asked `_locked_contact`, which additionally required
    `contact.engageable`. Measured on a self-play run: 99.0% of steps
    held a lock but only 31.8% held an engageable one, and the observation bit was
    true on ~0.44 steps/episode against 187.8 counted opportunities.
    """
    from bvr_marl_core.rl.environment.spaces.observation.own_state_builder import OwnStateBuilder

    designated = _Contact("designated")
    unit = _Unit()

    sync_designated_contact(unit, designated, 3)
    assert OwnStateBuilder._cue_target(unit) is designated
    assert unit.max_missiles_per_target == 3


def test_a_non_engageable_designation_still_reaches_the_observation():
    """`engageable` is a track-STATE predicate the launch path never applies.

    A coasting track drops out of {CONFIRMED, REACQUIRED} constantly, which is exactly
    the churn behind the seeker defect. Filtering on it here pinned the fire axis to
    P(fire)=0 on steps where the launch path would have taken the shot.
    """
    from bvr_marl_core.rl.environment.spaces.observation.own_state_builder import OwnStateBuilder

    coasting = _Contact("coasting", engageable=False)
    unit = _Unit()
    sync_designated_contact(unit, coasting, 2)

    assert OwnStateBuilder._cue_target(unit) is coasting


def test_no_designation_is_authoritative_not_a_fallback_to_the_lock():
    """An explicit "nothing designated" must not be papered over with a locked contact.

    The launch path would veto with `no_target_selected`, so the cues must agree.
    """
    from bvr_marl_core.rl.environment.spaces.observation.own_state_builder import OwnStateBuilder

    unit = _Unit()
    sync_designated_contact(unit, None, 2)

    assert OwnStateBuilder._cue_target(unit) is None


def test_unmirrored_unit_falls_back_to_the_locked_contact():
    """The reset observation is built before any action has run."""
    from bvr_marl_core.rl.environment.spaces.observation.own_state_builder import OwnStateBuilder

    unit = _Unit()  # no mirror at all
    assert not hasattr(unit, "designated_contact")
    # Falls through to _locked_contact, which tolerates a unit with no sensor.
    assert OwnStateBuilder._cue_target(unit) is None


def test_geometry_gates_are_never_silently_skipped():
    """A unit with no ``observation_helper`` must still have its geometry evaluated.

    ``gimbal_ok``, ``radar_range_ok`` and ``weapon_range_ok`` all default to True, so
    ``_helper_of`` returning None does not skip an optional refinement -- it reports a
    shot whose geometry nobody looked at. Aircraft units carry no ``observation_helper``
    attribute, so the launch path ran that way on every step while the observation path
    passed its own helper and ran with geometry.

    Measured over 3781 steps of trained-policy rollout before the fix:
    ``shot_opportunities`` 10.26% of steps against an observation bit of 2.06%, with all
    188 gate-level disagreements reading ``weapon_range_ok: launch=True obs=False``.
    After it: 2.30% vs 2.06%, and 74/74 agreement when both are evaluated at the same
    instant.
    """
    from bvr_marl_core.aircraft.systems.fire_feasibility import _helper_of

    unit = _Unit()
    assert not hasattr(unit, "observation_helper")

    helper = _helper_of(unit)
    assert helper is not None, "a missing helper silently disables every geometry gate"
    assert hasattr(helper, "get_geometry_kinematics")
    # Cached on the unit, so the hot loop builds it once.
    assert unit.observation_helper is helper
    assert _helper_of(unit) is helper


def test_observation_builders_do_not_cache_helpers_across_episodes():
    """A ``{unit.id: helper}`` cache outlives the unit it wraps.

    Unit ids repeat every episode while unit objects are rebuilt (verified in a live env:
    unit id 1 in both episodes, different Python objects). A builder-owned cache keyed on
    the id therefore returns a helper wrapping the PREVIOUS episode's dead aircraft from
    the second episode onward -- and training reuses env instances, so every episode after
    the first read its ownship cues off a corpse.
    """
    from bvr_marl_core.rl.environment.spaces.observation.missile_warning_builder import (
        MissileWarningBuilder,
    )
    from bvr_marl_core.rl.environment.spaces.observation.own_state_builder import OwnStateBuilder

    for cls in (OwnStateBuilder, MissileWarningBuilder):
        assert not hasattr(cls, "_obs_helpers"), f"{cls.__name__} kept an id-keyed cache"

    builder = OwnStateBuilder.__new__(OwnStateBuilder)
    assert not hasattr(builder, "_obs_helpers")

    episode_1 = _Unit()
    episode_1.id = 1
    episode_2 = _Unit()  # same id, new object -- what reset() produces
    episode_2.id = 1

    helper_1 = builder._get_obs_helper(episode_1)
    helper_2 = builder._get_obs_helper(episode_2)

    assert helper_1 is not helper_2, "helper leaked across the episode boundary"
    assert helper_1.aircraft is episode_1
    assert helper_2.aircraft is episode_2
    # Same unit, same instance: the helper is cached, just on the right owner.
    assert builder._get_obs_helper(episode_2) is helper_2


def test_helper_lookup_still_tolerates_a_unit_that_cannot_host_the_attribute():
    """Test doubles that reject attribute assignment keep the permissive behaviour."""
    from bvr_marl_core.aircraft.systems.fire_feasibility import _helper_of

    class _Slotted:
        __slots__ = ()

    assert _helper_of(_Slotted()) is None
    assert _helper_of(None) is None


def test_the_duplicate_predicates_are_gone():
    """Guards the unification: neither call site may re-hand-roll the conjunction."""
    from bvr_marl_core.rl.environment.spaces.action_space import weapon_firing

    assert not hasattr(weapon_firing.WeaponFiringHandler, "_sensor_lock_ok")
    assert not hasattr(weapon_firing.WeaponFiringHandler, "_sensor_fov_ok")

    from bvr_marl_core.aircraft.systems import observation_helper

    assert not hasattr(observation_helper.ObservationHelper, "_legacy_get_fire_feasibility")
