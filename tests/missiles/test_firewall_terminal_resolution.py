"""Firewall audit (item 2): the terminal seeker never resolves its physical target
by using an anonymous operational track ID as a simulator-roster key.

A WeaponTrack-guided (sensor-limited) missile holds a track ID drawn from a namespace
that is disjoint from simulator unit IDs. Resolving the physical unit the seeker
samples this tick must go through the evaluator-side weapon--target association
(``sim.evaluator_target_for_weapon``), never a roster lookup keyed by that ID -- the
two can collide numerically and pull an unrelated, or friendly, unit into the terminal
candidate set. These tests lock the invariant so a refactor cannot reintroduce a
track-id -> unit-id fallback on the operational path, and document that the only
id-keyed resolver is reachable exclusively on the oracle/direct-launch path.
"""

import inspect
from types import SimpleNamespace

from bvr_marl_core.missiles.missile import Missile


def _unit(uid, group="red"):
    return SimpleNamespace(
        id=uid,
        position=object(),
        group=group,
        is_destroyed=False,
        is_missile=False,
        is_countermeasure=False,
        is_non_engageable=False,
    )


def _stub_missile(**attrs):
    m = Missile.__new__(Missile)
    m.group = "blue"
    m.oracle_direct_launch = False
    m.radar = SimpleNamespace(get_locked_target=lambda: None)
    for key, value in attrs.items():
        setattr(m, key, value)
    return m


def _sim(association, departed=False):
    return SimpleNamespace(
        evaluator_target_for_weapon=lambda _m: association,
        evaluator_weapon_target_departed=lambda _m: departed,
        record_diagnostic=lambda *_: None,
    )


def test_operational_path_uses_evaluator_association_by_identity():
    associated = _unit(999)
    collider = _unit(7)  # id collides with the missile's anonymous track id below
    sim = _sim(associated)
    m = _stub_missile(designated_target_id=7)

    out = Missile._resolve_substep_sensor_target(m, sim, [collider, associated], True)

    assert out is associated  # identity match to the evaluator-associated unit
    assert out is not collider  # never the id-colliding roster unit


def test_operational_path_never_falls_back_to_roster_key():
    # No evaluator association this tick: the result is None, never a lookup of the
    # colliding roster unit by the missile's track id.
    collider = _unit(3)
    m = _stub_missile(designated_target_id=3)

    out = Missile._resolve_substep_sensor_target(m, _sim(None), [collider], True)

    assert out is None


def test_operational_association_must_be_in_engageable_roster():
    # The evaluator returns a unit that is not in the (pre-filtered) roster; the
    # identity check fails, so no candidate is produced -- value-equality is never used.
    stray = _unit(5)
    m = _stub_missile()

    assert Missile._resolve_substep_sensor_target(m, _sim(stray), [], True) is None


def test_oracle_path_is_the_only_one_that_keys_by_id():
    # Contrast: on the oracle/direct-launch path the locked id genuinely IS a unit id,
    # so resolution by id is legitimate there (and only there).
    target = _unit(42)
    other = _unit(43)
    m = _stub_missile(oracle_direct_launch=True, designated_target_id=42)

    out = Missile._resolve_substep_sensor_target(m, _sim(None), [other, target], False)

    assert out is target


def test_id_keyed_guidance_resolver_is_gated_to_non_operational_path():
    # The only resolver that keys sim.active_units by a target id is
    # _resolve_substep_guidance_target. Every call site in Missile.update must sit
    # behind an ``operational_track_only else`` guard (the oracle/direct-launch path).
    normalized = " ".join(inspect.getsource(Missile.update).split())
    needle = "_resolve_substep_guidance_target(sim)"
    idx = 0
    sites = 0
    while True:
        found = normalized.find(needle, idx)
        if found == -1:
            break
        sites += 1
        preceding = normalized[max(0, found - 80) : found]
        assert "operational_track_only else" in preceding, preceding
        idx = found + 1
    assert sites >= 1, "expected the id-keyed resolver to be called in Missile.update"
