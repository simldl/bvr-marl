"""tactical/lock_rate and fov_rate must measure SENSOR COVERAGE, not trigger pulls.

`termination.py` divides `episode_lock_ok_count` by the agent's total step count, so
`state["last_lock_ok"]` has to be written on EVERY step. It used to be written only on
the firing paths -- set from launch diagnostics on a successful shot, cleared on a
vetoed press -- so between trigger pulls it held a stale value. The metric was therefore
a sticky latch driven by shooting: an agent that never fired reported exactly 0.0 no
matter how well it held a lock, and the 0.18-0.25 stage floors were unreachable by
construction.
"""

from __future__ import annotations

import pytest

from bvr_marl_core.aircraft.systems.fire_feasibility import (
    own_radar_lock_ok,
    weapon_fov_ok,
)
from bvr_marl_core.rl.environment.spaces.action_space.weapon_firing import (
    WeaponFiringHandler,
)


class _Sensor:
    def __init__(self, locked_ids=(), lock_on_object=False):
        self._locked = set(locked_ids)
        self._lock_on_object = lock_on_object

    def get_locked_targets(self):
        return set(self._locked)

    def has_radar_lock(self, target):
        return self._lock_on_object


class _Weapons:
    def __init__(self, in_fov: bool):
        self._in_fov = in_fov

    def is_contact_in_fov(self, contact):
        return self._in_fov

    def is_target_in_fov(self, target):
        return self._in_fov


class _Unit:
    def __init__(self, locked_ids=(), in_fov=False, lock_on_object=False):
        self.sensor = _Sensor(locked_ids, lock_on_object)
        self.weapons = _Weapons(in_fov)


class _Target:
    """A plain (non-TacticalContact) target, i.e. the oracle/truth path."""

    id = "bandit-1"


def _contact(track_id="trk-1"):
    pytest.importorskip("numpy")
    from bvr_marl_core.domain.tactical_contact import TacticalContact

    # Only track_id is read by the code under test; build via __new__ so the test does
    # not depend on the full contact schema.
    contact = TacticalContact.__new__(TacticalContact)
    object.__setattr__(contact, "track_id", track_id)
    return contact


# -- lock ------------------------------------------------------------------


def test_lock_is_true_while_the_designated_contact_is_locked():
    unit = _Unit(locked_ids={"trk-1"})

    assert own_radar_lock_ok(unit, _contact("trk-1")) is True


def test_lock_is_false_for_a_contact_that_is_merely_tracked():
    unit = _Unit(locked_ids={"other"})

    assert own_radar_lock_ok(unit, _contact("trk-1")) is False


def test_lock_is_false_without_a_designated_target():
    assert own_radar_lock_ok(_Unit(locked_ids={"trk-1"}), None) is False


def test_lock_uses_has_radar_lock_on_the_truth_path():
    assert own_radar_lock_ok(_Unit(lock_on_object=True), _Target()) is True
    assert own_radar_lock_ok(_Unit(lock_on_object=False), _Target()) is False


def test_lock_is_false_rather_than_raising_when_the_unit_has_no_sensor():
    class _NoSensor:
        sensor = None
        weapons = _Weapons(True)

    assert own_radar_lock_ok(_NoSensor(), _contact()) is False


# -- fov -------------------------------------------------------------------


def test_fov_tracks_the_weapon_system_answer():
    assert weapon_fov_ok(_Unit(in_fov=True), _contact()) is True
    assert weapon_fov_ok(_Unit(in_fov=False), _contact()) is False


def test_fov_is_false_without_a_designated_target():
    # Note this differs from the GUN fov check, which returns True for no target so an
    # untargeted gun burst is not blamed on FOV. For the metric, "no target" is not
    # coverage.
    assert weapon_fov_ok(_Unit(in_fov=True), None) is False


# -- the property that was broken ------------------------------------------


def test_sensor_state_is_independent_of_whether_the_trigger_was_pulled():
    # The regression, stated directly: a locked, in-FOV agent that never fires must
    # still report coverage. Under the old code these flags were only written inside the
    # firing branches, so this read False forever and lock_rate was pinned at 0.0.
    unit = _Unit(locked_ids={"trk-1"}, in_fov=True)
    target = _contact("trk-1")

    assert own_radar_lock_ok(unit, target) is True
    assert weapon_fov_ok(unit, target) is True


def test_launch_gate_outcome_no_longer_shares_the_coverage_keys():
    # The launch-time diagnostics still exist, under their own keys, so shot-quality
    # analysis keeps its signal without overwriting per-step coverage.
    import inspect

    source = inspect.getsource(WeaponFiringHandler._fire_missile)

    assert "last_launch_lock_ok" in source
    assert "last_launch_fov_ok" in source
    assert 'state["last_lock_ok"]' not in source
    assert 'state["last_fov_ok"]' not in source
