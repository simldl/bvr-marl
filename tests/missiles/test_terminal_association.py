"""Terminal target resolution must never key the roster with an operational ID."""

from types import SimpleNamespace

from bvr_marl_core.missiles.missile import Missile

# The collision under test: an anonymous operational track ID that happens to
# equal a physical unit ID belonging to a different aircraft.
COLLIDING_ID = 7


def _missile(locked_target, *, oracle_direct_launch=False, designated_target_id=None):
    return SimpleNamespace(
        radar=SimpleNamespace(get_locked_target=lambda: locked_target),
        oracle_direct_launch=oracle_direct_launch,
        designated_target_id=designated_target_id,
    )


def _sim(evaluator_target, diagnostics):
    return SimpleNamespace(
        evaluator_target_for_weapon=lambda missile: evaluator_target,
        record_diagnostic=diagnostics.append,
    )


def _resolve(missile, sim, radar_targets, operational_track_only):
    return Missile._resolve_substep_sensor_target(
        missile, sim, radar_targets, operational_track_only
    )


def test_weapon_track_missile_ignores_unit_whose_id_equals_its_track_id():
    """A track/unit ID collision must not put an unrelated unit in the candidate set."""
    collider = SimpleNamespace(id=COLLIDING_ID)
    diagnostics = []

    resolved = _resolve(
        _missile(locked_target=COLLIDING_ID),
        _sim(evaluator_target=None, diagnostics=diagnostics),
        [collider],
        True,
    )

    assert resolved is None
    assert diagnostics == ["missile_terminal_association_missing"]


def test_weapon_track_missile_resolves_through_evaluator_association():
    intended = SimpleNamespace(id=42)
    collider = SimpleNamespace(id=COLLIDING_ID)

    resolved = _resolve(
        _missile(locked_target=COLLIDING_ID),
        _sim(evaluator_target=intended, diagnostics=[]),
        [collider, intended],
        True,
    )

    assert resolved is intended


def test_weapon_track_missile_rejects_association_outside_the_engageable_roster():
    """A destroyed or friendly evaluator target is not a valid terminal candidate."""
    departed = SimpleNamespace(id=42)

    resolved = _resolve(
        _missile(locked_target=COLLIDING_ID),
        _sim(evaluator_target=departed, diagnostics=[]),
        [SimpleNamespace(id=COLLIDING_ID)],
        True,
    )

    assert resolved is None


def test_weapon_track_missile_fails_closed_without_an_evaluator_registry():
    diagnostics = []
    sim = SimpleNamespace(record_diagnostic=diagnostics.append)

    resolved = _resolve(_missile(locked_target=COLLIDING_ID), sim, [SimpleNamespace(id=7)], True)

    assert resolved is None
    assert diagnostics == ["missile_terminal_association_missing"]


def test_oracle_direct_launch_still_resolves_by_physical_unit_id():
    """Outside the operational path locked_target really is a unit ID."""
    target = SimpleNamespace(id=COLLIDING_ID)

    resolved = _resolve(
        _missile(locked_target=COLLIDING_ID),
        _sim(evaluator_target=None, diagnostics=[]),
        [target],
        False,
    )

    assert resolved is target


def test_oracle_direct_launch_falls_back_to_the_designated_target():
    target = SimpleNamespace(id=99)

    resolved = _resolve(
        _missile(locked_target=None, oracle_direct_launch=True, designated_target_id=99),
        _sim(evaluator_target=None, diagnostics=[]),
        [target],
        False,
    )

    assert resolved is target
