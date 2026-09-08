"""Tests for AWACS/datalink weapons-lock sharing (``group_locked_target_ids``).

This is what lets a fighter launch beyond its own radar range on an AWACS-cued
(or wingman-cued) lock, flying the missile on datalink midcourse guidance.
"""

from __future__ import annotations

from bvr_marl_core.radar.core.data_link import DataLink


class _Position:
    def __init__(self, lat, lon, alt=10000.0):
        self.lat = lat
        self.lon = lon
        self.alt = alt


class _DL:
    def __init__(self, mode):
        self._mode = mode

    def get_mode(self):
        return self._mode


class _Radar:
    def __init__(self, mode="full", locks=None):
        self.data_link = _DL(mode)
        self._locks = set(locks or [])

    def get_locked_targets(self):
        return set(self._locks)


class _Fighter:
    def __init__(self, uid, group, mode="full", locks=None):
        self.id = uid
        self.group = group
        self.position = _Position(0.0, 0.0)
        self.is_missile = False
        self.radar = _Radar(mode, locks)


class _AWACS:
    def __init__(self, uid, group, lockable_ids):
        self.id = uid
        self.group = group
        self.position = _Position(0.0, 0.0)
        self.is_missile = False
        self.radar = _Radar("full")
        self._lockable = set(lockable_ids)

    def can_lock_target(self, target):  # AWACS locking cone
        return getattr(target, "id", None) in self._lockable


class _Enemy:
    def __init__(self, uid, non_engageable=False):
        self.id = uid
        self.group = "red"
        self.is_missile = False
        self.is_non_engageable = non_engageable
        self.position = _Position(1.0, 0.0)


class _Sim:
    def __init__(self, units):
        self.active_units = {u.id: u for u in units}


def test_awacs_lock_within_cone_is_shared():
    shooter = _Fighter(1, "blue")
    awacs = _AWACS(5, "blue", lockable_ids={3})  # AWACS locks bandit 3 (in cone)
    e3, e4 = _Enemy(3), _Enemy(4)  # bandit 4 outside the cone
    sim = _Sim([shooter, awacs, e3, e4])
    locked = DataLink.group_locked_target_ids(sim, shooter)
    assert 3 in locked  # AWACS-cued lock available to the shooter
    assert 4 not in locked  # outside the AWACS cone → not lockable


def test_wingman_radar_lock_is_shared():
    shooter = _Fighter(1, "blue")
    wingman = _Fighter(2, "blue", locks={4})  # wingman has own-radar lock on 4
    sim = _Sim([shooter, wingman, _Enemy(4)])
    assert 4 in DataLink.group_locked_target_ids(sim, shooter)


def test_no_lock_without_full_datalink():
    shooter = _Fighter(1, "blue")
    awacs = _AWACS(5, "blue", lockable_ids={3})
    awacs.radar.data_link = _DL("own")  # not full datalink → no sharing
    sim = _Sim([shooter, awacs, _Enemy(3)])
    assert DataLink.group_locked_target_ids(sim, shooter) == set()
