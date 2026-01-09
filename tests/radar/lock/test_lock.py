import pytest
from radar.lock.aircraft import AircraftLockController
from radar.lock.missile import MissileLockController

def test_aircraft_lock_multi():
    lock = AircraftLockController()
    lock.update_locks(['a', 'b'])
    # nach zwei confirms
    lock.update_locks(['a', 'b'])
    assert lock.is_locked('a')
    assert lock.is_locked('b')
    # verliert 'a'
    for _ in range(5):
        lock.update_locks(['b'])
    assert not lock.is_locked('a')
    assert lock.is_locked('b')

def test_missile_lock_single():
    lock = MissileLockController()
    lock.update_locks(['t1'])
    lock.update_locks(['t1'])
    assert lock.is_locked()
    assert lock.get_locked() == 't1'
    # verliert Lock nach 5 Misses
    for _ in range(5):
        lock.update_locks([])
    assert not lock.is_locked()
    assert lock.get_locked() is None

def test_lock_switch_missile():
    lock = MissileLockController()
    lock.update_locks(['a'])
    lock.update_locks(['a'])
    assert lock.get_locked() == 'a'
    for _ in range(5):
        lock.update_locks(['b'])
    assert lock.get_locked() == 'b'
