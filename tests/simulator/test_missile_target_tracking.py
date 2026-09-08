"""
Tests for simulator missile-target tracking registry.

Covers:
1. _missile_target_counts incremented on add_unit for missiles
2. _missile_target_counts decremented on remove_unit
3. reset_sim clears the registry
4. count_missiles_at_target returns correct counts
5. Non-missiles do not affect the registry
6. Retarget registry sync in GuidanceTargetProvider

Note: WeaponCooldowns.is_target_saturated tests live in the extension package,
because WeaponCooldowns is defined there and requires that package installed.
"""

import pytest

from bvr_marl_core.simulator.core.units import Position
from bvr_marl_core.simulator.simulator import Simulator

# ---------------------------------------------------------------------------
# Helpers
#
# These are explicit stubs rather than ``unittest.mock.Mock``. A bare Mock
# fabricates every attribute it is asked for, so weapon-identity probes
# (``weapon_track``, ``launch_contact_id``, ``missiles``) silently answer with a
# truthy Mock and a legacy stub gets misread as an operational one. Declaring the
# attributes keeps "this weapon has no weapon track" an assertable fact.
# ---------------------------------------------------------------------------


class _StubUnit:
    """A minimal unit the simulator registry can account for."""

    is_countermeasure = False
    is_missile = False
    # Legacy/oracle-path identity: no operational weapon track, no launch contact.
    weapon_track = None
    launch_contact_id = None
    launch_sensor_id = None
    launch_report_lineage = ()
    designated_target_id = None

    def __init__(self, group, target=None, unit_id=None):
        self.group = group
        self.target = target
        self.position = Position(45.0, 10.0, 5000.0)
        self.id = unit_id
        self.missiles = []
        # Kinematic surface the simulator reads when it records a unit trace.
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.speed = 300.0

    def update(self, tick_secs, sim):
        return []

    def substep_update(self, dt, sim):
        return []


class _StubMissile(_StubUnit):
    is_missile = True


def _make_target(target_id):
    return _StubUnit("red", unit_id=target_id)


def _make_aircraft(group):
    """Create a minimal stub aircraft (not a missile)."""
    return _StubUnit(group)


def _register_target(sim, target_id) -> _StubUnit:
    """Register a dummy aircraft in sim with the given id, return it."""
    ac = _make_aircraft("red")
    ac.id = target_id  # pre-set so add_unit uses it
    # add_unit normally assigns id; bypass by inserting directly
    sim.active_units[target_id] = ac
    return ac


def _make_missile(group, target, sim):
    """Create a minimal stub missile whose target is pre-registered in sim."""
    return _StubMissile(group, target=target)


# ---------------------------------------------------------------------------
# Tests: Simulator._missile_target_counts via add_unit / remove_unit
# ---------------------------------------------------------------------------


class TestMissileTargetCountsRegistry:
    @pytest.fixture
    def sim(self):
        return Simulator(tick_secs=1.0)

    def test_initial_registry_empty(self, sim):
        assert sim._missile_target_counts == {}

    def test_add_missile_increments_count(self, sim):
        tgt = _register_target(sim, 42)
        m = _make_missile("blue", tgt, sim)
        sim.add_unit(m)
        assert sim.count_missiles_at_target("blue", 42) == 1

    def test_add_two_missiles_same_target(self, sim):
        tgt = _register_target(sim, 42)
        m1 = _make_missile("blue", tgt, sim)
        m2 = _make_missile("blue", tgt, sim)
        sim.add_unit(m1)
        sim.add_unit(m2)
        assert sim.count_missiles_at_target("blue", 42) == 2

    def test_teammates_count_same_contact_by_shared_report_lineage(self, sim):
        missile = _StubMissile("blue")
        missile.id = 10
        missile.launch_sensor_id = 1
        missile.launch_contact_id = "local-a"
        missile.launch_report_lineage = ((7, 101), (8, 202))
        sim.active_units[missile.id] = missile

        assert (
            sim.count_missiles_at_contact("blue", 2, "local-b", report_lineage=((8, 202), (9, 303)))
            == 1
        )
        assert (
            sim.count_missiles_at_contact("blue", 2, "unrelated", report_lineage=((9, 303),)) == 0
        )

    def test_add_missiles_different_targets(self, sim):
        tgt1 = _register_target(sim, 1)
        tgt2 = _register_target(sim, 2)
        m1 = _make_missile("blue", tgt1, sim)
        m2 = _make_missile("blue", tgt2, sim)
        sim.add_unit(m1)
        sim.add_unit(m2)
        assert sim.count_missiles_at_target("blue", 1) == 1
        assert sim.count_missiles_at_target("blue", 2) == 1

    def test_add_missiles_different_groups(self, sim):
        tgt = _register_target(sim, 99)
        m1 = _make_missile("blue", tgt, sim)
        m2 = _make_missile("red", tgt, sim)
        sim.add_unit(m1)
        sim.add_unit(m2)
        assert sim.count_missiles_at_target("blue", 99) == 1
        assert sim.count_missiles_at_target("red", 99) == 1

    def test_non_missile_does_not_affect_registry(self, sim):
        a = _make_aircraft("blue")
        sim.add_unit(a)
        assert sim._missile_target_counts == {}

    def test_remove_missile_decrements_count(self, sim):
        tgt = _register_target(sim, 42)
        m = _make_missile("blue", tgt, sim)
        sim.add_unit(m)
        assert sim.count_missiles_at_target("blue", 42) == 1
        sim.remove_unit(m.id)
        assert sim.count_missiles_at_target("blue", 42) == 0

    def test_remove_one_of_two_missiles(self, sim):
        tgt = _register_target(sim, 42)
        m1 = _make_missile("blue", tgt, sim)
        m2 = _make_missile("blue", tgt, sim)
        sim.add_unit(m1)
        sim.add_unit(m2)
        sim.remove_unit(m1.id)
        assert sim.count_missiles_at_target("blue", 42) == 1

    def test_remove_last_missile_clears_key(self, sim):
        tgt = _register_target(sim, 42)
        m = _make_missile("blue", tgt, sim)
        sim.add_unit(m)
        sim.remove_unit(m.id)
        # Key should be gone, not just zeroed
        assert 42 not in sim._missile_target_counts.get("blue", {})

    def test_count_unknown_group_returns_zero(self, sim):
        assert sim.count_missiles_at_target("blue", 99) == 0

    def test_count_unknown_target_returns_zero(self, sim):
        tgt = _register_target(sim, 1)
        m = _make_missile("blue", tgt, sim)
        sim.add_unit(m)
        assert sim.count_missiles_at_target("blue", 999) == 0

    def test_reset_sim_clears_registry(self, sim):
        tgt = _register_target(sim, 42)
        m = _make_missile("blue", tgt, sim)
        sim.add_unit(m)
        assert sim.count_missiles_at_target("blue", 42) == 1
        sim.reset_sim(units={})
        assert sim._missile_target_counts == {}

    def test_missile_with_no_target_does_not_raise(self, sim):
        tgt = _register_target(sim, 42)
        m = _make_missile("blue", tgt, sim)
        m.target = None  # No target — bypass validation
        # Directly inject to bypass add_unit target check
        m.id = sim._next_unit_id
        sim._next_unit_id += 1
        sim.active_units[m.id] = m
        # Registry should be unaffected (can't increment without target)
        assert sim._missile_target_counts == {}

    def test_missile_with_no_group_does_not_raise(self, sim):
        tgt = _register_target(sim, 42)
        m = _make_missile("blue", tgt, sim)
        m.group = None
        # Bypass add_unit target validation by injecting directly
        m.id = sim._next_unit_id
        sim._next_unit_id += 1
        sim.active_units[m.id] = m
        assert sim._missile_target_counts == {}


# ---------------------------------------------------------------------------
# Tests: retarget registry sync in GuidanceTargetProvider
# ---------------------------------------------------------------------------


class TestRetargetRegistrySync:
    """
    Verify that when GuidanceTargetProvider switches a missile's target,
    the simulator _missile_target_counts registry is updated correctly.
    These tests directly exercise the registry sync logic that target_provider uses.
    """

    def test_retarget_updates_counts(self):
        """When missile switches target, old count decrements and new count increments."""
        sim = Simulator(tick_secs=1.0)

        # Two potential targets — must be registered in sim
        target_a = _register_target(sim, 100)
        target_b = _register_target(sim, 200)
        target_a_id = 100
        target_b_id = 200

        # Missile initially targeting A
        missile = _make_missile("blue", target_a, sim)
        sim.add_unit(missile)
        assert sim.count_missiles_at_target("blue", target_a_id) == 1
        assert sim.count_missiles_at_target("blue", target_b_id) == 0

        # Simulate a retarget: apply the registry sync logic as in target_provider
        old_target_id = getattr(getattr(missile, "target", None), "id", None)
        new_target_id = target_b_id

        if old_target_id != new_target_id:
            sim.resync_missile_target(missile.group, old_target_id, new_target_id)

        missile.target = target_b

        # After retarget: A should be 0, B should be 1
        assert sim.count_missiles_at_target("blue", target_a_id) == 0
        assert sim.count_missiles_at_target("blue", target_b_id) == 1

    def test_two_missiles_retarget_one(self):
        """Two missiles target A; one retargets to B — A count drops to 1."""
        sim = Simulator(tick_secs=1.0)

        target_a = _register_target(sim, 100)
        target_b = _register_target(sim, 200)
        target_a_id = 100
        target_b_id = 200

        m1 = _make_missile("blue", target_a, sim)
        m2 = _make_missile("blue", target_a, sim)
        sim.add_unit(m1)
        sim.add_unit(m2)
        assert sim.count_missiles_at_target("blue", target_a_id) == 2

        # Retarget m2 from A to B
        sim.resync_missile_target("blue", target_a_id, target_b_id)
        m2.target = target_b

        assert sim.count_missiles_at_target("blue", target_a_id) == 1
        assert sim.count_missiles_at_target("blue", target_b_id) == 1
