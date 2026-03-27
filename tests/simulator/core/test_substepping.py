"""
Comprehensive tests for simulator.core.substepping module.

Tests the Substepper class for adaptive substepping behavior,
missile-target engagement scenarios, and integration methods.
"""

import math
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from air_to_air_rl.simulator.core.substepping import SubstepConfig, Substepper


class TestSubstepConfig:
    """Test SubstepConfig dataclass."""

    def test_default_initialization(self):
        """Test SubstepConfig with default values."""
        config = SubstepConfig()

        assert config.engage_distance_km == 2.0
        assert config.min_substeps == 4
        assert config.max_substeps == 16
        assert config.safety_travel_per_substep_m == 250.0
        assert config.min_dt == 1e-3
        assert not config.physics_only

    def test_custom_initialization(self):
        """Test SubstepConfig with custom values."""
        config = SubstepConfig(
            engage_distance_km=5.0,
            min_substeps=8,
            max_substeps=32,
            safety_travel_per_substep_m=100.0,
            min_dt=1e-4,
            physics_only=True,
        )

        assert config.engage_distance_km == 5.0
        assert config.min_substeps == 8
        assert config.max_substeps == 32
        assert config.safety_travel_per_substep_m == 100.0
        assert config.min_dt == 1e-4
        assert config.physics_only

    def test_dataclass_equality(self):
        """Test SubstepConfig equality comparison."""
        config1 = SubstepConfig(engage_distance_km=3.0, min_substeps=6)
        config2 = SubstepConfig(engage_distance_km=3.0, min_substeps=6)
        config3 = SubstepConfig(engage_distance_km=4.0, min_substeps=6)

        assert config1 == config2
        assert config1 != config3


class MockUnit:
    """Helper class for creating mock units with configurable attributes."""

    def __init__(
        self,
        unit_id: int = 1,
        is_missile: bool = False,
        is_destroyed: bool = False,
        target=None,
        locked_target=None,
        speed: float = 100.0,
        unit_type: str = "",
        class_name: str = "MockUnit",
    ):
        self.id = unit_id
        self.is_missile = is_missile
        self.is_destroyed = is_destroyed
        self.target = target
        self.locked_target = locked_target
        self.speed = speed
        self.unit_type = unit_type
        self.__class__.__name__ = class_name

        # Add common attributes that might be accessed
        self.vel_enu = None
        self.physics_step_called = False
        self.substep_update_called = False

    def physics_step(self, dt: float, sim):
        """Mock physics step method."""
        self.physics_step_called = True
        return []

    def substep_update(self, dt: float, sim):
        """Mock substep update method."""
        self.substep_update_called = True
        return []


class TestSubstepperInitialization:
    """Test Substepper initialization."""

    def test_default_initialization(self):
        """Test initialization with default config."""
        substepper = Substepper()

        assert isinstance(substepper.cfg, SubstepConfig)
        assert substepper.cfg.engage_distance_km == 2.0
        assert substepper.cfg.min_substeps == 4

    def test_custom_config_initialization(self):
        """Test initialization with custom config."""
        custom_config = SubstepConfig(engage_distance_km=5.0, min_substeps=8)
        substepper = Substepper(custom_config)

        assert substepper.cfg is custom_config
        assert substepper.cfg.engage_distance_km == 5.0
        assert substepper.cfg.min_substeps == 8

    def test_none_config_creates_default(self):
        """Test initialization with None config creates default."""
        substepper = Substepper(None)

        assert isinstance(substepper.cfg, SubstepConfig)
        assert substepper.cfg.engage_distance_km == 2.0


class TestSubstepperIsMissile:
    """Test _is_missile static method."""

    def test_destroyed_unit_not_missile(self):
        """Test that destroyed units are not considered missiles."""
        unit = MockUnit(is_missile=True, is_destroyed=True)
        assert Substepper._is_missile(unit) is False

    def test_is_missile_flag_true(self):
        """Test unit with is_missile=True."""
        unit = MockUnit(is_missile=True)
        assert Substepper._is_missile(unit) is True

    def test_is_missile_flag_false(self):
        """Test unit with is_missile=False."""
        unit = MockUnit(is_missile=False)
        assert Substepper._is_missile(unit) is False

    def test_unit_type_missile(self):
        """Test unit with unit_type='missile'."""
        unit = MockUnit(unit_type="missile")
        assert Substepper._is_missile(unit) is True

    def test_unit_type_missile_case_insensitive(self):
        """Test unit_type matching is case insensitive."""
        unit = MockUnit(unit_type="MISSILE")
        assert Substepper._is_missile(unit) is True

    def test_class_name_ends_with_missile(self):
        """Test class name ending with 'missile'."""
        unit = MockUnit(class_name="AIM120Missile")
        assert Substepper._is_missile(unit) is True

    def test_class_name_missile_case_insensitive(self):
        """Test class name matching is case insensitive."""
        unit = MockUnit(class_name="AIM120MISSILE")
        assert Substepper._is_missile(unit) is True

    def test_non_missile_unit(self):
        """Test non-missile unit returns False."""
        unit = MockUnit(is_missile=False, unit_type="aircraft", class_name="F16")
        assert Substepper._is_missile(unit) is False

    def test_unit_without_attributes(self):
        """Test unit without missile-related attributes."""
        unit = Mock(spec=[])  # No attributes
        assert Substepper._is_missile(unit) is False


class TestSubstepperCollectCandidates:
    """Test _collect_candidates method."""

    @patch("air_to_air_rl.simulator.core.substepping.units_distance_km")
    def test_no_missiles(self, mock_distance):
        """Test with no missiles in units."""
        substepper = Substepper()
        units = [MockUnit(is_missile=False), MockUnit(is_missile=False)]

        result = substepper._collect_candidates(units)

        assert result == []
        mock_distance.assert_not_called()

    @patch("air_to_air_rl.simulator.core.substepping.units_distance_km")
    def test_missile_without_target(self, mock_distance):
        """Test missile without target."""
        substepper = Substepper()
        missile = MockUnit(is_missile=True, target=None, locked_target=None)
        units = [missile]

        result = substepper._collect_candidates(units)

        assert result == []
        mock_distance.assert_not_called()

    @patch("air_to_air_rl.simulator.core.substepping.units_distance_km")
    def test_missile_with_destroyed_target(self, mock_distance):
        """Test missile with destroyed target."""
        substepper = Substepper()
        target = MockUnit(is_destroyed=True)
        missile = MockUnit(is_missile=True, target=target)
        units = [missile, target]

        result = substepper._collect_candidates(units)

        assert result == []
        mock_distance.assert_not_called()

    @patch("air_to_air_rl.simulator.core.substepping.units_distance_km")
    def test_missile_target_within_range(self, mock_distance):
        """Test missile with target within engage distance."""
        mock_distance.return_value = 1.5  # Within default 2.0 km

        substepper = Substepper()
        target = MockUnit()
        missile = MockUnit(is_missile=True, target=target)
        units = [missile, target]

        result = substepper._collect_candidates(units)

        assert result == [(missile, target)]
        mock_distance.assert_called_once_with(missile, target)

    @patch("air_to_air_rl.simulator.core.substepping.units_distance_km")
    def test_missile_target_beyond_range(self, mock_distance):
        """Test missile with target beyond engage distance."""
        mock_distance.return_value = 3.0  # Beyond default 2.0 km

        substepper = Substepper()
        target = MockUnit()
        missile = MockUnit(is_missile=True, target=target)
        units = [missile, target]

        result = substepper._collect_candidates(units)

        assert result == []
        mock_distance.assert_called_once_with(missile, target)

    @patch("air_to_air_rl.simulator.core.substepping.units_distance_km")
    def test_locked_target_preferred_over_target(self, mock_distance):
        """Test that locked_target is preferred over target."""
        mock_distance.return_value = 1.0

        substepper = Substepper()
        target = MockUnit(unit_id=1)
        locked_target = MockUnit(unit_id=2)
        missile = MockUnit(is_missile=True, target=target, locked_target=locked_target)
        units = [missile, target, locked_target]

        result = substepper._collect_candidates(units)

        assert result == [(missile, locked_target)]
        mock_distance.assert_called_once_with(missile, locked_target)

    @patch("air_to_air_rl.simulator.core.substepping.units_distance_km")
    def test_multiple_missile_target_pairs(self, mock_distance):
        """Test multiple missile-target pairs."""
        mock_distance.side_effect = [1.0, 1.5, 3.0]  # First two within range

        substepper = Substepper()
        target1 = MockUnit(unit_id=1)
        target2 = MockUnit(unit_id=2)
        target3 = MockUnit(unit_id=3)
        missile1 = MockUnit(unit_id=11, is_missile=True, target=target1)
        missile2 = MockUnit(unit_id=12, is_missile=True, target=target2)
        missile3 = MockUnit(unit_id=13, is_missile=True, target=target3)
        units = [missile1, missile2, missile3, target1, target2, target3]

        result = substepper._collect_candidates(units)

        assert len(result) == 2
        assert (missile1, target1) in result
        assert (missile2, target2) in result
        assert (missile3, target3) not in result

    @patch("air_to_air_rl.simulator.core.substepping.units_distance_km")
    def test_distance_calculation_exception(self, mock_distance):
        """Test handling of distance calculation exceptions."""
        mock_distance.side_effect = Exception("Distance calculation failed")

        substepper = Substepper()
        target = MockUnit()
        missile = MockUnit(is_missile=True, target=target)
        units = [missile, target]

        result = substepper._collect_candidates(units)

        assert result == []  # Should ignore pair on exception

    @patch("air_to_air_rl.simulator.core.substepping.units_distance_km")
    def test_custom_engage_distance(self, mock_distance):
        """Test with custom engage distance."""
        mock_distance.return_value = 4.0

        config = SubstepConfig(engage_distance_km=5.0)
        substepper = Substepper(config)
        target = MockUnit()
        missile = MockUnit(is_missile=True, target=target)
        units = [missile, target]

        result = substepper._collect_candidates(units)

        assert result == [(missile, target)]  # 4.0 < 5.0, should be included


class TestSubstepperRunTickWithSubsteps:
    """Test run_tick_with_substeps method."""

    @patch("air_to_air_rl.simulator.core.substepping.Substepper._collect_candidates")
    def test_no_candidates(self, mock_collect):
        """Test with no missile-target candidates."""
        mock_collect.return_value = []

        substepper = Substepper()
        sim = Mock()

        result = substepper.run_tick_with_substeps(sim, 1.0)

        assert result == []
        mock_collect.assert_called_once()

    @patch("air_to_air_rl.simulator.core.substepping.Substepper._collect_candidates")
    def test_with_ccd_hit_calc_available(self, mock_collect):
        """Test with CCD hit calculator available."""
        substepper = Substepper()

        missile = MockUnit(is_missile=True)
        target = MockUnit()
        mock_collect.return_value = [(missile, target)]

        # Mock simulator with nested CCD structure
        mock_ccd = Mock()
        mock_hit_calc = Mock()
        mock_hit_calc.check_and_maybe_hit.return_value = (True, target, 0.5)
        mock_ccd.hit_calc = mock_hit_calc
        mock_ccd.on_hit = Mock()

        sim = Mock()
        sim.ccd = mock_ccd
        sim.active_units = {missile.id: missile, target.id: target}

        result = substepper.run_tick_with_substeps(sim, 1.0)

        assert result == []
        mock_hit_calc.check_and_maybe_hit.assert_called_once_with(
            missile=missile, potential_targets=[target], tick_secs=1.0
        )
        mock_ccd.on_hit.assert_called_once_with(missile, target, 0.5, sim)

    @patch("air_to_air_rl.simulator.core.substepping.Substepper._collect_candidates")
    def test_with_direct_hit_calc_available(self, mock_collect):
        """Test with direct hit calculator on sim."""
        substepper = Substepper()

        missile = MockUnit(is_missile=True)
        target = MockUnit()
        mock_collect.return_value = [(missile, target)]

        # Mock simulator with direct hit_calc
        mock_hit_calc = Mock()
        mock_hit_calc.check_and_maybe_hit.return_value = (False, None, None)

        sim = Mock()
        sim.ccd = None
        sim.hit_calc = mock_hit_calc

        result = substepper.run_tick_with_substeps(sim, 1.0)

        assert result == []
        mock_hit_calc.check_and_maybe_hit.assert_called_once()

    @patch("air_to_air_rl.simulator.core.substepping.Substepper._collect_candidates")
    def test_ccd_miss_no_hit_callback(self, mock_collect):
        """Test CCD miss doesn't call on_hit."""
        missile = MockUnit(is_missile=True)
        target = MockUnit()
        mock_collect.return_value = [(missile, target)]

        mock_ccd = Mock()
        mock_hit_calc = Mock()
        mock_hit_calc.check_and_maybe_hit.return_value = (False, None, None)
        mock_ccd.hit_calc = mock_hit_calc
        mock_ccd.on_hit = Mock()

        sim = Mock()
        sim.ccd = mock_ccd

        substepper = Substepper()
        result = substepper.run_tick_with_substeps(sim, 1.0)

        assert result == []
        mock_ccd.on_hit.assert_not_called()

    @patch(
        "air_to_air_rl.simulator.core.substepping.Substepper._integrate_temporarily_with_restore"
    )
    @patch("air_to_air_rl.simulator.core.substepping.Substepper._collect_candidates")
    def test_fallback_to_integration(self, mock_collect, mock_integrate):
        """Test fallback to conservative integration when no hit calc."""
        missile = MockUnit(is_missile=True)
        target = MockUnit()
        mock_collect.return_value = [(missile, target)]

        sim = Mock()
        sim.ccd = None
        sim.hit_calc = None

        substepper = Substepper()
        result = substepper.run_tick_with_substeps(sim, 1.0)

        assert result == []
        mock_integrate.assert_called_once_with(sim, missile, target, 1.0)

    @patch("air_to_air_rl.simulator.core.substepping.Substepper._collect_candidates")
    def test_multiple_pairs_all_processed(self, mock_collect):
        """Test that multiple missile-target pairs are all processed."""
        missile1 = MockUnit(unit_id=1, is_missile=True)
        target1 = MockUnit(unit_id=2)
        missile2 = MockUnit(unit_id=3, is_missile=True)
        target2 = MockUnit(unit_id=4)
        mock_collect.return_value = [(missile1, target1), (missile2, target2)]

        mock_hit_calc = Mock()
        mock_hit_calc.check_and_maybe_hit.return_value = (False, None, None)

        sim = Mock()
        sim.ccd = None
        sim.hit_calc = mock_hit_calc

        substepper = Substepper()
        result = substepper.run_tick_with_substeps(sim, 1.0)

        assert result == []
        assert mock_hit_calc.check_and_maybe_hit.call_count == 2


class TestSubstepperRelativeSpeedEstimation:
    """Test _estimate_relative_speed_mps method."""

    def test_with_vel_enu_attributes(self):
        """Test speed estimation using vel_enu attributes."""
        substepper = Substepper()

        unit_a = MockUnit()
        unit_a.vel_enu = [100.0, 50.0, 10.0]  # velocity vector
        unit_b = MockUnit()
        unit_b.vel_enu = [80.0, 30.0, 5.0]  # different velocity

        result = substepper._estimate_relative_speed_mps(unit_a, unit_b)

        # Expected: sqrt((100-80)² + (50-30)² + (10-5)²) = sqrt(400+400+25) = sqrt(825)
        expected = math.sqrt(825)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_with_vel_enu_same_velocity(self):
        """Test speed estimation with identical velocities."""
        substepper = Substepper()

        unit_a = MockUnit()
        unit_a.vel_enu = [100.0, 50.0, 10.0]
        unit_b = MockUnit()
        unit_b.vel_enu = [100.0, 50.0, 10.0]  # Same velocity

        result = substepper._estimate_relative_speed_mps(unit_a, unit_b)

        assert result == pytest.approx(0.0, abs=1e-10)

    def test_with_vel_enu_exception_fallback(self):
        """Test fallback to speed difference when vel_enu calculation fails."""
        substepper = Substepper()

        unit_a = MockUnit(speed=200.0)
        unit_a.vel_enu = "invalid"  # Will cause exception
        unit_b = MockUnit(speed=150.0)
        unit_b.vel_enu = [80.0, 30.0, 5.0]

        result = substepper._estimate_relative_speed_mps(unit_a, unit_b)

        # Should fallback to |200 - 150| = 50
        assert result == pytest.approx(50.0, abs=1e-10)

    def test_without_vel_enu_attributes(self):
        """Test speed estimation using only speed attributes."""
        substepper = Substepper()

        unit_a = MockUnit(speed=250.0)
        unit_b = MockUnit(speed=180.0)

        result = substepper._estimate_relative_speed_mps(unit_a, unit_b)

        assert result == pytest.approx(70.0, abs=1e-10)  # |250 - 180|

    def test_with_zero_speeds(self):
        """Test speed estimation with zero speeds."""
        substepper = Substepper()

        unit_a = MockUnit(speed=0.0)
        unit_b = MockUnit(speed=0.0)

        result = substepper._estimate_relative_speed_mps(unit_a, unit_b)

        assert result == pytest.approx(0.0, abs=1e-10)

    def test_with_none_speeds(self):
        """Test speed estimation with None speeds."""
        substepper = Substepper()

        unit_a = MockUnit()
        unit_a.speed = None
        unit_b = MockUnit(speed=100.0)

        result = substepper._estimate_relative_speed_mps(unit_a, unit_b)

        assert result == pytest.approx(100.0, abs=1e-10)  # |0 - 100|

    def test_with_string_speed_conversion(self):
        """Test speed conversion from string."""
        substepper = Substepper()

        unit_a = MockUnit()
        unit_a.speed = "200.5"  # String that can be converted to float
        unit_b = MockUnit(speed=100.0)

        result = substepper._estimate_relative_speed_mps(unit_a, unit_b)

        assert result == pytest.approx(100.5, abs=1e-10)


class TestSubstepperEstimateSubsteps:
    """Test _estimate_substeps method."""

    def test_very_low_relative_speed(self):
        """Test with very low relative speed returns min substeps."""
        substepper = Substepper()

        missile = MockUnit(speed=1e-7)
        target = MockUnit(speed=0.0)

        with patch.object(substepper, "_estimate_relative_speed_mps", return_value=1e-7):
            result = substepper._estimate_substeps(missile, target, 1.0)

            assert result == substepper.cfg.min_substeps

    def test_normal_relative_speed(self):
        """Test normal relative speed calculation."""
        config = SubstepConfig(safety_travel_per_substep_m=100.0, min_substeps=2, max_substeps=20)
        substepper = Substepper(config)

        missile = MockUnit(speed=500.0)  # 500 m/s
        target = MockUnit(speed=300.0)  # 300 m/s

        with patch.object(
            substepper, "_estimate_relative_speed_mps", return_value=200.0
        ):  # |500-300|
            result = substepper._estimate_substeps(missile, target, 2.0)

            # Expected: ceil((200 * 2.0) / 100.0) = ceil(4.0) = 4
            assert result == 4

    def test_high_relative_speed_clamped_to_max(self):
        """Test high relative speed is clamped to max substeps."""
        config = SubstepConfig(safety_travel_per_substep_m=50.0, min_substeps=2, max_substeps=10)
        substepper = Substepper(config)

        missile = MockUnit(speed=1000.0)
        target = MockUnit(speed=0.0)

        with patch.object(substepper, "_estimate_relative_speed_mps", return_value=1000.0):
            result = substepper._estimate_substeps(missile, target, 2.0)

            # Expected: ceil((1000 * 2.0) / 50.0) = ceil(40) = 40, clamped to 10
            assert result == config.max_substeps

    def test_low_calculated_substeps_clamped_to_min(self):
        """Test low calculated substeps clamped to min."""
        config = SubstepConfig(safety_travel_per_substep_m=1000.0, min_substeps=5, max_substeps=20)
        substepper = Substepper(config)

        missile = MockUnit(speed=100.0)
        target = MockUnit(speed=90.0)

        with patch.object(substepper, "_estimate_relative_speed_mps", return_value=10.0):
            result = substepper._estimate_substeps(missile, target, 1.0)

            # Expected: ceil((10 * 1.0) / 1000.0) = ceil(0.01) = 1, clamped to 5
            assert result == config.min_substeps

    def test_exact_substeps_calculation(self):
        """Test exact substeps calculation without clamping."""
        config = SubstepConfig(safety_travel_per_substep_m=200.0, min_substeps=1, max_substeps=50)
        substepper = Substepper(config)

        missile = MockUnit(speed=300.0)
        target = MockUnit(speed=100.0)

        with patch.object(substepper, "_estimate_relative_speed_mps", return_value=200.0):
            result = substepper._estimate_substeps(missile, target, 1.5)

            # Expected: ceil((200 * 1.5) / 200.0) = ceil(1.5) = 2
            assert result == 2


class TestSubstepperPhysicsOnlyStep:
    """Test _physics_only_step method."""

    def test_with_physics_step_method(self):
        """Test unit with physics_step method."""
        substepper = Substepper()
        unit = Mock()
        unit.physics_step = Mock()
        sim = Mock()

        substepper._physics_only_step(sim, unit, 0.5)

        unit.physics_step.assert_called_once_with(0.5, sim)

    def test_with_update_physics_method(self):
        """Test unit with update_physics method (fallback)."""
        substepper = Substepper()
        unit = Mock()
        unit.physics_step = None  # Not available
        unit.update_physics = Mock()
        sim = Mock()

        substepper._physics_only_step(sim, unit, 1.0)

        unit.update_physics.assert_called_once_with(1.0, sim)

    def test_without_physics_methods_raises_error(self):
        """Test unit without physics methods raises RuntimeError."""
        substepper = Substepper()
        unit = Mock()
        unit.physics_step = None
        unit.update_physics = None
        sim = Mock()

        with pytest.raises(RuntimeError) as exc_info:
            substepper._physics_only_step(sim, unit, 1.0)

        assert "has no physics_step/update_physics" in str(exc_info.value)

    def test_physics_step_preferred_over_update_physics(self):
        """Test that physics_step is preferred over update_physics."""
        substepper = Substepper()
        unit = Mock()
        unit.physics_step = Mock()
        unit.update_physics = Mock()
        sim = Mock()

        substepper._physics_only_step(sim, unit, 0.2)

        unit.physics_step.assert_called_once_with(0.2, sim)
        unit.update_physics.assert_not_called()


class TestSubstepperIntegrateTemporarilyWithRestore:
    """Test _integrate_temporarily_with_restore method."""

    def test_without_ccd_hooks_returns_early(self):
        """Test method returns early without required CCD hooks."""
        substepper = Substepper()
        sim = Mock()
        sim.ccd = None
        missile = MockUnit(is_missile=True)
        target = MockUnit()

        # Should return without error
        substepper._integrate_temporarily_with_restore(sim, missile, target, 1.0)

        # No assertions needed - just shouldn't crash

    def test_with_incomplete_ccd_hooks_returns_early(self):
        """Test method returns early with incomplete CCD hooks."""
        substepper = Substepper()
        sim = Mock()
        sim.ccd = Mock(spec=["capture_state"])  # Only has capture_state, missing other hooks
        sim.ccd.capture_state = Mock()

        missile = MockUnit(is_missile=True)
        target = MockUnit()

        # Should return without error
        substepper._integrate_temporarily_with_restore(sim, missile, target, 1.0)

    def test_without_begin_state_returns_early(self):
        """Test method returns early when begin state is not available."""
        substepper = Substepper()
        sim = Mock()
        sim.ccd = Mock()
        sim.ccd.capture_state = Mock(side_effect=[{}, {}])  # Current states
        sim.ccd.get_begin_state = Mock(return_value=None)  # No begin state
        sim.ccd.apply_state = Mock()

        missile = MockUnit(is_missile=True)
        target = MockUnit()

        substepper._integrate_temporarily_with_restore(sim, missile, target, 1.0)

        # Should capture states but return early
        assert sim.ccd.capture_state.call_count == 2
        sim.ccd.apply_state.assert_not_called()

    @patch("air_to_air_rl.simulator.core.substepping.Substepper._estimate_substeps")
    def test_full_integration_with_physics_only(self, mock_estimate):
        """Test full integration with physics_only=True."""
        mock_estimate.return_value = 4

        config = SubstepConfig(physics_only=True, min_dt=0.1)
        substepper = Substepper(config)

        # Mock CCD with all required methods
        sim = Mock()
        sim.ccd = Mock()
        sim.ccd.capture_state = Mock(side_effect=[{"missile": True}, {"target": True}])
        sim.ccd.get_begin_state = Mock(
            side_effect=[{"begin_missile": True}, {"begin_target": True}]
        )
        sim.ccd.apply_state = Mock()
        sim.ccd.hit_calc = None  # Ensure CCD doesn't have hit_calc either
        sim.hit_calc = None

        # Mock units with physics_step
        missile = MockUnit(is_missile=True)
        missile.physics_step = Mock()
        target = MockUnit()
        target.physics_step = Mock()

        substepper._integrate_temporarily_with_restore(sim, missile, target, 1.0)

        # Should integrate in substeps (4 substeps with dt=0.25 each)
        expected_calls = 4
        assert missile.physics_step.call_count == expected_calls
        assert target.physics_step.call_count == expected_calls

        # Should restore states at the end
        expected_restore_calls = [call(missile, {"missile": True}), call(target, {"target": True})]
        sim.ccd.apply_state.assert_has_calls(expected_restore_calls)

    @patch("air_to_air_rl.simulator.core.substepping.Substepper._estimate_substeps")
    def test_full_integration_with_substep_update(self, mock_estimate):
        """Test full integration with substep_update methods."""
        mock_estimate.return_value = 2

        config = SubstepConfig(physics_only=False)
        substepper = Substepper(config)

        # Mock CCD
        sim = Mock()
        sim.ccd = Mock()
        sim.ccd.capture_state = Mock(side_effect=[{}, {}])
        sim.ccd.get_begin_state = Mock(side_effect=[{}, {}])
        sim.ccd.apply_state = Mock()
        sim.ccd.hit_calc = None
        sim.hit_calc = None

        # Mock units with substep_update methods
        missile = MockUnit(is_missile=True)
        missile.substep_update = Mock()
        target = MockUnit()
        target.substep_update = Mock()

        substepper._integrate_temporarily_with_restore(sim, missile, target, 1.0)

        # Should call substep_update methods
        assert missile.substep_update.call_count == 2
        assert target.substep_update.call_count == 2

        # Check call signature (dt, sim) for substep_update
        missile.substep_update.assert_called_with(0.5, sim)
        target.substep_update.assert_called_with(0.5, sim)

    @patch("air_to_air_rl.simulator.core.substepping.Substepper._estimate_substeps")
    def test_integration_with_hit_detection(self, mock_estimate):
        """Test integration with hit detection during substeps."""
        mock_estimate.return_value = 2

        substepper = Substepper()

        # Mock CCD and hit calculator
        sim = Mock()
        sim.ccd = Mock()
        sim.ccd.capture_state = Mock(side_effect=[{}, {}])
        sim.ccd.get_begin_state = Mock(side_effect=[{}, {}])
        sim.ccd.apply_state = Mock()
        sim.ccd.on_hit = Mock()
        sim.ccd.hit_calc = None  # No CCD hit_calc, use sim.hit_calc instead

        # Mock hit calculator with hit on second substep
        hit_calc = Mock()
        hit_calc.check_and_maybe_hit = Mock(
            side_effect=[
                (False, None, None),  # First substep: no hit
                (True, "target", 0.5),  # Second substep: hit
            ]
        )
        sim.hit_calc = hit_calc

        # Mock units
        missile = MockUnit(is_missile=True)
        missile.physics_step = Mock()
        target = MockUnit()
        target.physics_step = Mock()

        substepper._integrate_temporarily_with_restore(sim, missile, target, 1.0)

        # Should break after hit detected
        assert hit_calc.check_and_maybe_hit.call_count == 2
        sim.ccd.on_hit.assert_called_once_with(missile, "target", 0.5, sim)

        # Should still restore states after hit
        sim.ccd.apply_state.assert_called()

    @patch("air_to_air_rl.simulator.core.substepping.Substepper._estimate_substeps")
    def test_min_dt_enforcement(self, mock_estimate):
        """Test that min_dt is enforced during integration."""
        mock_estimate.return_value = 2000  # Very high substeps

        config = SubstepConfig(min_dt=0.1, physics_only=True)  # Minimum dt, physics only
        substepper = Substepper(config)

        # Mock CCD
        sim = Mock()
        sim.ccd = Mock()
        sim.ccd.capture_state = Mock(side_effect=[{}, {}])
        sim.ccd.get_begin_state = Mock(side_effect=[{}, {}])
        sim.ccd.apply_state = Mock()
        sim.ccd.hit_calc = None
        sim.hit_calc = None

        # Mock units
        missile = MockUnit(is_missile=True)
        missile.physics_step = Mock()
        target = MockUnit()
        target.physics_step = Mock()

        substepper._integrate_temporarily_with_restore(sim, missile, target, 1.0)

        # Should use min_dt=0.1, so 10 steps for 1.0 second
        expected_calls = 10
        assert missile.physics_step.call_count == expected_calls
        assert target.physics_step.call_count == expected_calls


class TestSubstepperIntegration:
    """Integration tests for complete Substepper scenarios."""

    @patch("air_to_air_rl.simulator.core.substepping.units_distance_km")
    def test_complete_substep_scenario(self, mock_distance):
        """Test complete substepping scenario from start to finish."""
        mock_distance.return_value = 1.5  # Within engage range

        # Create substepper with custom config
        config = SubstepConfig(engage_distance_km=2.0, min_substeps=2, max_substeps=8)
        substepper = Substepper(config)

        # Create missile and target
        missile = MockUnit(unit_id=1, is_missile=True, speed=300.0)
        target = MockUnit(unit_id=2, speed=200.0)
        missile.target = target
        missile.physics_step = Mock()
        target.physics_step = Mock()

        # Mock simulator with hit calculator
        sim = Mock()
        sim.active_units = {1: missile, 2: target}
        sim.ccd = None  # No CCD, use direct hit_calc
        sim.hit_calc = Mock()
        sim.hit_calc.check_and_maybe_hit.return_value = (False, None, None)

        # Run substepping
        result = substepper.run_tick_with_substeps(sim, 1.0)

        # Verify process completed
        assert result == []
        mock_distance.assert_called_once_with(missile, target)
        sim.hit_calc.check_and_maybe_hit.assert_called_once()

    @patch("air_to_air_rl.simulator.core.substepping.units_distance_km")
    def test_multiple_missiles_different_outcomes(self, mock_distance):
        """Test multiple missiles with different engagement outcomes."""
        mock_distance.side_effect = [1.0, 1.5, 3.0]  # First two in range

        substepper = Substepper()

        # Create multiple missile-target pairs
        missile1 = MockUnit(unit_id=1, is_missile=True)
        target1 = MockUnit(unit_id=2)
        missile1.target = target1

        missile2 = MockUnit(unit_id=3, is_missile=True)
        target2 = MockUnit(unit_id=4)
        missile2.target = target2

        missile3 = MockUnit(unit_id=5, is_missile=True)
        target3 = MockUnit(unit_id=6)
        missile3.target = target3

        # Mock simulator
        sim = Mock()
        sim.active_units = {
            1: missile1,
            2: target1,
            3: missile2,
            4: target2,
            5: missile3,
            6: target3,
        }

        # Mock hit calculator with different outcomes
        hit_calc = Mock()
        hit_calc.check_and_maybe_hit.side_effect = [
            (True, target1, 0.3),  # missile1 hits
            (False, None, None),  # missile2 misses
            # missile3 not processed (out of range)
        ]
        sim.hit_calc = hit_calc

        # Mock CCD for hit handling
        sim.ccd = Mock()
        sim.ccd.hit_calc = None  # No CCD hit_calc, use sim.hit_calc instead
        sim.ccd.on_hit = Mock()

        substepper.run_tick_with_substeps(sim, 1.0)

        # Only first two missiles should be processed
        assert hit_calc.check_and_maybe_hit.call_count == 2
        sim.ccd.on_hit.assert_called_once_with(missile1, target1, 0.3, sim)

    def test_edge_case_zero_tick_time(self):
        """Test edge case with zero tick time."""
        substepper = Substepper()
        sim = Mock()
        sim.active_units = {}

        result = substepper.run_tick_with_substeps(sim, 0.0)

        assert result == []

    def test_edge_case_very_small_tick_time(self):
        """Test edge case with very small tick time."""
        config = SubstepConfig(min_dt=1e-6)
        substepper = Substepper(config)
        sim = Mock()
        sim.active_units = {}

        result = substepper.run_tick_with_substeps(sim, 1e-9)  # Smaller than min_dt

        assert result == []
