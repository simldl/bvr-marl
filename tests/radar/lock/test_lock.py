import pytest

from air_to_air_rl.radar.lock.aircraft import AircraftLockController
from air_to_air_rl.radar.lock.missile import MissileLockController


def test_aircraft_lock_multi():
    lock = AircraftLockController()
    lock.update_locks(["a", "b"])
    # nach zwei confirms
    lock.update_locks(["a", "b"])
    assert lock.is_locked("a")
    assert lock.is_locked("b")
    # verliert 'a'
    for _ in range(5):
        lock.update_locks(["b"])
    assert not lock.is_locked("a")
    assert lock.is_locked("b")


def test_missile_lock_single():
    lock = MissileLockController()
    lock.update_locks(["t1"])
    lock.update_locks(["t1"])
    assert lock.is_locked()
    assert lock.get_locked() == "t1"
    # verliert Lock nach 5 Misses
    for _ in range(5):
        lock.update_locks([])
    assert not lock.is_locked()
    assert lock.get_locked() is None


def test_lock_switch_missile():
    lock = MissileLockController()
    lock.update_locks(["a"])
    lock.update_locks(["a"])
    assert lock.get_locked() == "a"
    for _ in range(5):
        lock.update_locks(["b"])
    assert lock.get_locked() == "b"


def test_ordering_stability():
    """A1+A2: update_locks preserves input order; two controllers given the same sorted input
    produce identical _lock_state key order (mimics _update_lock_ctrl which pre-sorts).
    """
    sorted_ids = sorted(["z", "m", "a"])  # ['a', 'm', 'z']

    lock1 = MissileLockController()
    lock1.update_locks(sorted_ids)
    lock1.update_locks(sorted_ids)

    lock2 = MissileLockController()
    lock2.update_locks(sorted_ids)
    lock2.update_locks(sorted_ids)

    ids1 = list(lock1._lock_state.keys())
    ids2 = list(lock2._lock_state.keys())
    assert ids1 == ids2 == sorted_ids, f"Expected {sorted_ids}, got {ids1}"


def test_hysteresis_prevents_marginal_switch():
    """B2: force_active sets target; marginally-better competitor does NOT trigger switch."""
    lock = MissileLockController()
    # Confirm both targets
    lock.update_locks(["t1", "t2"])
    lock.update_locks(["t1", "t2"])
    # Explicitly pin to t1
    lock.force_active("t1")
    assert lock.get_locked() == "t1"
    # Continued updates with both targets present — fallback path keeps t1
    lock.update_locks(["t1", "t2"])
    assert lock.get_locked() == "t1", "Hysteresis should keep t1 when no explicit switch is made"


def test_switch_on_large_improvement():
    """B1: force_active switches when called with clearly better target."""
    lock = MissileLockController()
    lock.update_locks(["t1", "t2"])
    lock.update_locks(["t1", "t2"])
    lock.force_active("t1")
    assert lock.get_locked() == "t1"
    # Radar decides t2 is better and calls force_active
    lock.force_active("t2")
    assert lock.get_locked() == "t2"
