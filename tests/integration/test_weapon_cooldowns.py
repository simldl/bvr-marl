"""
Tests for WeaponCooldowns.is_target_saturated (behavior–core integration).

These tests were split from bvr-marl-core/tests/simulator/test_missile_target_tracking.py
because WeaponCooldowns lives in bvr_marl_core and requires bvr_marl_core installed.
The pure-simulator registry tests remain in bvr-marl-core.
"""

from unittest.mock import Mock

import pytest

from bvr_marl_core.rl.environment.spaces.action_space.automation.weapon_cooldowns import (
    WeaponCooldowns,
)
from bvr_marl_core.simulator.core.units import Position
from bvr_marl_core.simulator.simulator import Simulator

# ---------------------------------------------------------------------------
# Helpers (mirrors bvr-marl-core/tests/simulator/test_missile_target_tracking.py)
# ---------------------------------------------------------------------------


def _make_target(target_id):
    t = Mock()
    t.id = target_id
    return t


def _make_aircraft(group):
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
    ac = _make_aircraft("red")
    ac.id = target_id
    sim.active_units[target_id] = ac
    return ac


def _make_missile(group, target, sim):
    m = Mock()
    m.is_missile = True
    m.is_countermeasure = False
    m.group = group
    m.target = target
    m.position = Position(45.0, 10.0, 5000.0)
    m.id = None
    m.update = Mock(return_value=[])
    m.substep_update = Mock(return_value=[])
    return m


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
