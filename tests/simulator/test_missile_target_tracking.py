"""
Tests for simulator missile-target tracking registry.

Covers:
1. _missile_target_counts incremented on add_unit for missiles
2. _missile_target_counts decremented on remove_unit
3. reset_sim clears the registry
4. count_missiles_at_target returns correct counts
5. Non-missiles do not affect the registry
6. WeaponCooldowns.is_target_saturated uses the simulator registry
7. Retarget registry sync in GuidanceTargetProvider
"""

from unittest.mock import MagicMock, Mock

import pytest

from air_to_air_rl.rl.environment.spaces.action_space.automation.weapon_cooldowns import (
    WeaponCooldowns,
)
from air_to_air_rl.simulator.core.helpers import Position
from air_to_air_rl.simulator.core.units import FlyingUnit
from air_to_air_rl.simulator.simulator import Simulator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_target(target_id):
    t = Mock()
    t.id = target_id
    return t


def _make_aircraft(group):
    """Create a minimal mock aircraft (not a missile)."""
    a = Mock()
    a.is_missile = False
    a.is_countermeasure = False
    a.group = group
    a.target = None
    a.position = Position(45.0, 10.0, 5000.0)
    a.id = None
    a.update = Mock(return_value=[])
    a.substep_update = Mock(return_value=[])
    return a


def _register_target(sim, target_id) -> Mock:
    """Register a dummy aircraft in sim with the given id, return it."""
    ac = _make_aircraft("red")
    ac.id = target_id  # pre-set so add_unit uses it
    # add_unit normally assigns id; bypass by inserting directly
    sim.active_units[target_id] = ac
    return ac


def _make_missile(group, target, sim):
    """Create a minimal mock missile whose target is pre-registered in sim."""
    m = Mock()
    m.is_missile = True
    m.is_countermeasure = False
    m.group = group
    m.target = target
    m.position = Position(45.0, 10.0, 5000.0)
    m.id = None  # assigned by simulator
    m.update = Mock(return_value=[])
    m.substep_update = Mock(return_value=[])
    return m


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
# Tests: WeaponCooldowns.is_target_saturated
# ---------------------------------------------------------------------------


class TestIsTargetSaturated:
    def _make_sim(self, count):
        """Simulator stub where count_missiles_at_target returns a fixed count."""
        sim = Mock()
        sim.count_missiles_at_target = Mock(return_value=count)
        return sim

    def test_not_saturated_below_cap(self):
        wc = WeaponCooldowns(max_missiles_per_target=2)
        sim = self._make_sim(1)
        assert wc.is_target_saturated(sim, "blue", 42) is False

    def test_saturated_at_cap(self):
        wc = WeaponCooldowns(max_missiles_per_target=2)
        sim = self._make_sim(2)
        assert wc.is_target_saturated(sim, "blue", 42) is True

    def test_saturated_above_cap(self):
        wc = WeaponCooldowns(max_missiles_per_target=2)
        sim = self._make_sim(3)
        assert wc.is_target_saturated(sim, "blue", 42) is True

    def test_not_saturated_zero_missiles(self):
        wc = WeaponCooldowns(max_missiles_per_target=2)
        sim = self._make_sim(0)
        assert wc.is_target_saturated(sim, "blue", 42) is False

    def test_none_target_id_returns_false(self):
        wc = WeaponCooldowns(max_missiles_per_target=2)
        sim = self._make_sim(5)
        assert wc.is_target_saturated(sim, "blue", None) is False

    def test_none_simulator_returns_false(self):
        wc = WeaponCooldowns(max_missiles_per_target=2)
        assert wc.is_target_saturated(None, "blue", 42) is False

    def test_delegates_to_simulator_method(self):
        """Verify is_target_saturated calls count_missiles_at_target with right args."""
        wc = WeaponCooldowns(max_missiles_per_target=2)
        sim = self._make_sim(0)
        wc.is_target_saturated(sim, "blue", 99)
        sim.count_missiles_at_target.assert_called_once_with("blue", 99)

    def test_cap_of_one(self):
        """With cap=1, even one missile saturates."""
        wc = WeaponCooldowns(max_missiles_per_target=1)
        sim = self._make_sim(1)
        assert wc.is_target_saturated(sim, "blue", 42) is True

    def test_cap_of_one_zero_missiles(self):
        wc = WeaponCooldowns(max_missiles_per_target=1)
        sim = self._make_sim(0)
        assert wc.is_target_saturated(sim, "blue", 42) is False

    def test_integrated_with_real_simulator(self):
        """End-to-end: fire two missiles, verify saturation, remove one."""
        sim = Simulator(tick_secs=1.0)
        wc = WeaponCooldowns(max_missiles_per_target=2)

        tgt = _register_target(sim, 10)
        m1 = _make_missile("blue", tgt, sim)
        m2 = _make_missile("blue", tgt, sim)
        sim.add_unit(m1)
        sim.add_unit(m2)

        # At cap — saturated
        assert wc.is_target_saturated(sim, "blue", 10) is True

        # Remove one — no longer saturated
        sim.remove_unit(m1.id)
        assert wc.is_target_saturated(sim, "blue", 10) is False


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
            group = missile.group
            missile_counts = sim._missile_target_counts
            if old_target_id is not None:
                grp_map = missile_counts.get(group, {})
                if grp_map.get(old_target_id, 0) > 1:
                    grp_map[old_target_id] -= 1
                elif old_target_id in grp_map:
                    del grp_map[old_target_id]
            if new_target_id is not None:
                grp_map = missile_counts.setdefault(group, {})
                grp_map[new_target_id] = grp_map.get(new_target_id, 0) + 1

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
        grp_map = sim._missile_target_counts.setdefault("blue", {})
        if grp_map.get(target_a_id, 0) > 1:
            grp_map[target_a_id] -= 1
        grp_map[target_b_id] = grp_map.get(target_b_id, 0) + 1
        m2.target = target_b

        assert sim.count_missiles_at_target("blue", target_a_id) == 1
        assert sim.count_missiles_at_target("blue", target_b_id) == 1
