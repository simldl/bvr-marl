"""
Comprehensive tests for simulator.core.hit_event_helpers module.

Tests the CCDConfig, MissileCCDManager, and related callback functions
for missile collision detection and event handling.
"""

from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from air_to_air_rl.simulator.core.hit_calculation import HitParams
from air_to_air_rl.simulator.core.hit_event_helpers import (
    CCDConfig,
    MissileCCDManager,
    default_on_hit,
    default_target_selector,
)


class TestCCDConfig:
    """Test CCDConfig dataclass."""

    def test_default_initialization(self):
        """Test CCDConfig with default values."""
        config = CCDConfig()

        assert config.fuse_radius_m == 500.0
        assert config.target_radius_m == 0.0
        assert config.max_consider_range_m == 50000.0
        assert config.within_lock_gate_m == 10000.0

    def test_custom_initialization(self):
        """Test CCDConfig with custom values."""
        config = CCDConfig(
            fuse_radius_m=200.0,
            target_radius_m=50.0,
            max_consider_range_m=30000.0,
            within_lock_gate_m=5000.0,
        )

        assert config.fuse_radius_m == 200.0
        assert config.target_radius_m == 50.0
        assert config.max_consider_range_m == 30000.0
        assert config.within_lock_gate_m == 5000.0

    def test_dataclass_equality(self):
        """Test CCDConfig equality comparison."""
        config1 = CCDConfig(fuse_radius_m=300.0, within_lock_gate_m=8000.0)
        config2 = CCDConfig(fuse_radius_m=300.0, within_lock_gate_m=8000.0)
        config3 = CCDConfig(fuse_radius_m=400.0, within_lock_gate_m=8000.0)

        assert config1 == config2
        assert config1 != config3


class TestDefaultOnHit:
    """Test default_on_hit callback function."""

    def test_successful_hit_removes_both_units(self):
        """Test that successful hit removes both missile and target."""
        missile = Mock()
        missile.id = 1
        target = Mock()
        target.id = 2

        sim = Mock()
        sim.active_units = {1: missile, 2: target}
        sim.remove_unit = Mock()
        sim.log_event = Mock()

        default_on_hit(missile, target, 0.5, sim)

        # Should remove both units
        expected_calls = [call(2), call(1)]  # target first, then missile
        sim.remove_unit.assert_has_calls(expected_calls)

        # Should log two destruction events
        assert sim.log_event.call_count == 2

    def test_hit_with_target_not_in_active_units(self):
        """Test hit when target is not in active units."""
        missile = Mock()
        missile.id = 1
        target = Mock()
        target.id = 2

        sim = Mock()
        sim.active_units = {1: missile}  # Target not in active units
        sim.remove_unit = Mock()
        sim.log_event = Mock()

        default_on_hit(missile, target, 0.3, sim)

        # Should only remove missile
        sim.remove_unit.assert_called_once_with(1)

        # Should still try to log both events
        assert sim.log_event.call_count == 2

    def test_hit_with_missile_not_in_active_units(self):
        """Test hit when missile is not in active units."""
        missile = Mock()
        missile.id = 1
        target = Mock()
        target.id = 2

        sim = Mock()
        sim.active_units = {2: target}  # Missile not in active units
        sim.remove_unit = Mock()
        sim.log_event = Mock()

        default_on_hit(missile, target, 0.7, sim)

        # Should only remove target
        sim.remove_unit.assert_called_once_with(2)

        # Should still log events
        assert sim.log_event.call_count == 2

    def test_hit_with_neither_unit_in_active_units(self):
        """Test hit when neither unit is in active units."""
        missile = Mock()
        missile.id = 1
        target = Mock()
        target.id = 2

        sim = Mock()
        sim.active_units = {}  # Neither unit in active units
        sim.remove_unit = Mock()
        sim.log_event = Mock()

        default_on_hit(missile, target, 1.0, sim)

        # Should not remove any units
        sim.remove_unit.assert_not_called()

        # Should still log events
        assert sim.log_event.call_count == 2

    def test_event_logging_content(self):
        """Test that appropriate destruction events are logged."""
        from air_to_air_rl.simulator.core.events import UnitDestroyedEvent

        missile = Mock()
        missile.id = 1
        target = Mock()
        target.id = 2

        sim = Mock()
        sim.active_units = {1: missile, 2: target}
        sim.remove_unit = Mock()
        sim.log_event = Mock()

        with patch(
            "air_to_air_rl.simulator.core.hit_event_helpers.UnitDestroyedEvent"
        ) as mock_event_class:
            mock_event_class.return_value = "mock_event"

            default_on_hit(missile, target, 0.5, sim)

            # Should create two UnitDestroyedEvent instances
            expected_calls = [
                call(sim, missile, target),  # Target destroyed by missile
                call(sim, missile, missile),  # Missile destroys itself
            ]
            mock_event_class.assert_has_calls(expected_calls)

            # Should log both events
            sim.log_event.assert_has_calls([call("mock_event"), call("mock_event")])


class TestDefaultTargetSelector:
    """Test default_target_selector callback function."""

    def test_with_locked_target_only(self):
        """Test selector with only locked_target available."""
        target = Mock()
        missile = Mock()
        missile.locked_target = target
        # No 'target' attribute

        result = default_target_selector(missile)

        assert result is target

    def test_with_target_only(self):
        """Test selector with only target available."""
        target = Mock()
        missile = Mock()
        missile.target = target
        missile.locked_target = None

        result = default_target_selector(missile)

        assert result is target

    def test_with_both_targets_prefers_locked(self):
        """Test that locked_target is preferred over target."""
        locked_target = Mock()
        regular_target = Mock()
        missile = Mock()
        missile.locked_target = locked_target
        missile.target = regular_target

        result = default_target_selector(missile)

        assert result is locked_target

    def test_with_no_targets(self):
        """Test selector with no targets."""
        missile = Mock()
        missile.locked_target = None
        missile.target = None

        result = default_target_selector(missile)

        assert result is None

    def test_with_missing_target_attributes(self):
        """Test selector with missing target attributes."""
        missile = Mock(spec=[])  # No attributes

        result = default_target_selector(missile)

        assert result is None


class MockUnit:
    """Helper class for creating mock units with configurable attributes."""

    def __init__(
        self,
        unit_id: int = 1,
        is_missile: bool = False,
        is_destroyed: bool = False,
        target=None,
        locked_target=None,
        unit_type: str = "",
        class_name: str = "MockUnit",
    ):
        self.id = unit_id
        self.is_missile = is_missile
        self.is_destroyed = is_destroyed
        self.target = target
        self.locked_target = locked_target
        self.unit_type = unit_type
        self.__class__.__name__ = class_name

        # Position attributes
        self.position = Mock()
        self.position.lat = 45.0
        self.position.lon = -123.0
        self.position.alt = 1000.0

        # Create a copy method that returns a proper copy
        def copy_position():
            pos_copy = Mock()
            pos_copy.lat = self.position.lat
            pos_copy.lon = self.position.lon
            pos_copy.alt = self.position.alt
            return pos_copy

        self.position.copy = copy_position

        # Flight attributes
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.speed = 100.0
        self.desired_yaw_deg = 0.0
        self.desired_pitch_deg = 0.0


class TestMissileCCDManagerInitialization:
    """Test MissileCCDManager initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        manager = MissileCCDManager()

        assert isinstance(manager.cfg, CCDConfig)
        assert manager.cfg.fuse_radius_m == 500.0
        assert manager.on_hit == default_on_hit
        assert manager.target_selector == default_target_selector
        assert len(manager._engaged_once) == 0
        assert len(manager._begin_states) == 0

    def test_custom_config_initialization(self):
        """Test initialization with custom CCDConfig."""
        custom_config = CCDConfig(fuse_radius_m=300.0, within_lock_gate_m=8000.0)
        manager = MissileCCDManager(custom_config)

        assert manager.cfg is custom_config
        assert manager.cfg.fuse_radius_m == 300.0
        assert manager.cfg.within_lock_gate_m == 8000.0

    def test_custom_hit_params_initialization(self):
        """Test initialization with custom HitParams."""
        custom_hit_params = HitParams(fuse_radius_m=200.0, target_radius_m=25.0)
        manager = MissileCCDManager(hit_params=custom_hit_params)

        assert manager.hit_calc.params is custom_hit_params
        assert manager.hit_calc.params.fuse_radius_m == 200.0
        assert manager.hit_calc.params.target_radius_m == 25.0

    def test_custom_callbacks_initialization(self):
        """Test initialization with custom callback functions."""
        custom_on_hit = Mock()
        custom_target_selector = Mock()

        manager = MissileCCDManager(on_hit=custom_on_hit, target_selector=custom_target_selector)

        assert manager.on_hit is custom_on_hit
        assert manager.target_selector is custom_target_selector

    def test_hit_params_derived_from_config(self):
        """Test that HitParams are derived from CCDConfig when not provided."""
        config = CCDConfig(fuse_radius_m=400.0, target_radius_m=30.0, max_consider_range_m=60000.0)
        manager = MissileCCDManager(config)

        hit_params = manager.hit_calc.params
        assert hit_params.fuse_radius_m == 400.0
        assert hit_params.target_radius_m == 30.0
        assert hit_params.max_consider_range_m == 60000.0


class TestMissileCCDManagerSnapshot:
    """Test begin_tick_snapshot method."""

    def test_snapshot_empty_units(self):
        """Test snapshot with empty units."""
        manager = MissileCCDManager()

        manager.begin_tick_snapshot([])

        assert len(manager._begin_states) == 0

    def test_snapshot_single_unit(self):
        """Test snapshot with single unit."""
        manager = MissileCCDManager()
        unit = MockUnit(unit_id=1)

        with patch.object(manager.hit_calc, "begin_tick_snapshot") as mock_hit_snapshot:
            manager.begin_tick_snapshot([unit])

            mock_hit_snapshot.assert_called_once_with([unit])
            assert 1 in manager._begin_states
            assert isinstance(manager._begin_states[1], dict)

    def test_snapshot_multiple_units(self):
        """Test snapshot with multiple units."""
        manager = MissileCCDManager()
        units = [MockUnit(unit_id=i) for i in range(1, 4)]

        with patch.object(manager.hit_calc, "begin_tick_snapshot") as mock_hit_snapshot:
            manager.begin_tick_snapshot(units)

            mock_hit_snapshot.assert_called_once_with(units)
            assert len(manager._begin_states) == 3
            for i in range(1, 4):
                assert i in manager._begin_states

    def test_snapshot_unit_without_position(self):
        """Test snapshot with unit without position attribute."""
        manager = MissileCCDManager()
        unit = Mock(spec=["id"])  # Only has id attribute, no position
        unit.id = 1

        manager.begin_tick_snapshot([unit])

        assert 1 not in manager._begin_states

    def test_snapshot_clears_previous_states(self):
        """Test that snapshot clears previous begin states."""
        manager = MissileCCDManager()
        manager._begin_states[999] = {"old": "data"}  # Old data

        unit = MockUnit(unit_id=1)
        manager.begin_tick_snapshot([unit])

        assert 999 not in manager._begin_states
        assert 1 in manager._begin_states

    def test_snapshot_calls_hit_calc_snapshot(self):
        """Test that snapshot calls hit_calc.begin_tick_snapshot."""
        manager = MissileCCDManager()
        units = [MockUnit(unit_id=1), MockUnit(unit_id=2)]

        with patch.object(manager.hit_calc, "begin_tick_snapshot") as mock_hit_snapshot:
            manager.begin_tick_snapshot(units)

            mock_hit_snapshot.assert_called_once_with(units)


class TestMissileCCDManagerStateMethods:
    """Test state capture, get, and apply methods."""

    def test_capture_state_basic(self):
        """Test basic state capture."""
        manager = MissileCCDManager()
        unit = MockUnit(unit_id=1)
        unit.yaw_deg = 45.0
        unit.pitch_deg = 10.0
        unit.roll_deg = -5.0
        unit.speed = 200.0

        state = manager.capture_state(unit)

        assert state["yaw"] == 45.0
        assert state["pitch"] == 10.0
        assert state["roll"] == -5.0
        assert state["speed"] == 200.0
        # Position should be a copy with the same values
        assert state["pos"].lat == unit.position.lat
        assert state["pos"].lon == unit.position.lon
        assert state["pos"].alt == unit.position.alt

    def test_capture_state_with_desired_values(self):
        """Test state capture with desired control values."""
        manager = MissileCCDManager()
        unit = MockUnit(unit_id=1)
        unit.yaw_deg = 45.0
        unit.desired_yaw_deg = 60.0
        unit.pitch_deg = 10.0
        unit.desired_pitch_deg = 15.0

        state = manager.capture_state(unit)

        assert state["dyaw"] == 60.0
        assert state["dpitch"] == 15.0

    def test_capture_state_position_copy(self):
        """Test that position is copied when possible."""
        manager = MissileCCDManager()
        unit = MockUnit(unit_id=1)
        unit.position.copy = Mock(return_value="copied_position")

        state = manager.capture_state(unit)

        unit.position.copy.assert_called_once()
        assert state["pos"] == "copied_position"

    def test_capture_state_position_without_copy(self):
        """Test position handling when copy method unavailable."""
        manager = MissileCCDManager()
        unit = MockUnit(unit_id=1)
        unit.position = "no_copy_method"  # No copy method

        state = manager.capture_state(unit)

        assert state["pos"] == "no_copy_method"

    def test_capture_state_missing_attributes(self):
        """Test state capture with missing attributes."""
        manager = MissileCCDManager()
        unit = Mock(spec=["position"])  # Only has position, no yaw_deg, pitch_deg, etc.
        unit.position = Mock()
        unit.position.copy = Mock(return_value=unit.position)

        state = manager.capture_state(unit)

        assert state["yaw"] == 0.0  # Default value
        assert state["pitch"] == 0.0  # Default value
        assert state["roll"] == 0.0  # Default value
        assert state["speed"] == 0.0  # Default value

    def test_get_begin_state_existing(self):
        """Test getting existing begin state."""
        manager = MissileCCDManager()
        stored_state = {"test": "data"}
        manager._begin_states[1] = stored_state

        unit = MockUnit(unit_id=1)
        result = manager.get_begin_state(unit)

        assert result is stored_state

    def test_get_begin_state_missing(self):
        """Test getting non-existent begin state."""
        manager = MissileCCDManager()
        unit = MockUnit(unit_id=1)

        result = manager.get_begin_state(unit)

        assert result is None

    def test_apply_state_complete(self):
        """Test applying complete state."""
        manager = MissileCCDManager()
        unit = MockUnit(unit_id=1)

        # Mock position
        pos_mock = Mock()
        pos_mock.lat = 50.0
        pos_mock.lon = -120.0
        pos_mock.alt = 2000.0

        state = {
            "pos": pos_mock,
            "yaw": 90.0,
            "pitch": 15.0,
            "roll": -10.0,
            "speed": 250.0,
            "dyaw": 100.0,
            "dpitch": 20.0,
        }

        manager.apply_state(unit, state)

        # Check position was applied
        assert unit.position.lat == 50.0
        assert unit.position.lon == -120.0
        assert unit.position.alt == 2000.0

        # Check attitude and speed
        assert unit.yaw_deg == 90.0
        assert unit.pitch_deg == 15.0
        assert unit.roll_deg == -10.0
        assert unit.speed == 250.0

        # Check desired values
        assert unit.desired_yaw_deg == 100.0
        assert unit.desired_pitch_deg == 20.0

    def test_apply_state_none(self):
        """Test applying None state does nothing."""
        manager = MissileCCDManager()
        unit = MockUnit(unit_id=1)
        original_yaw = unit.yaw_deg

        manager.apply_state(unit, None)

        assert unit.yaw_deg == original_yaw  # Should be unchanged

    def test_apply_state_missing_attributes(self):
        """Test applying state when unit lacks attributes."""
        manager = MissileCCDManager()
        unit = Mock()
        # Unit has no attitude or speed attributes

        state = {"yaw": 45.0, "pitch": 10.0, "speed": 200.0}

        # Should not raise exception
        manager.apply_state(unit, state)

    def test_apply_state_partial(self):
        """Test applying partial state."""
        manager = MissileCCDManager()
        unit = MockUnit(unit_id=1)
        unit.yaw_deg = 0.0
        unit.pitch_deg = 0.0

        # Only apply yaw, not pitch
        state = {"yaw": 90.0}

        manager.apply_state(unit, state)

        assert unit.yaw_deg == 90.0
        assert unit.pitch_deg == 0.0  # Should be unchanged


class TestMissileCCDManagerUtilities:
    """Test utility methods of MissileCCDManager."""

    def test_uid_with_id_attribute(self):
        """Test _uid with unit that has id attribute."""
        unit = MockUnit(unit_id=42)

        result = MissileCCDManager._uid(unit)

        assert result == 42

    def test_uid_without_id_attribute(self):
        """Test _uid with unit without id attribute."""
        unit = Mock(spec=[])  # No id attribute

        result = MissileCCDManager._uid(unit)

        assert result == id(unit)  # Should use Python id()

    def test_is_active_missile_destroyed(self):
        """Test _is_active_missile with destroyed unit."""
        unit = MockUnit(is_missile=True, is_destroyed=True)

        result = MissileCCDManager._is_active_missile(unit)

        assert result is False

    def test_is_active_missile_flag_true(self):
        """Test _is_active_missile with is_missile=True."""
        unit = MockUnit(is_missile=True, is_destroyed=False)

        result = MissileCCDManager._is_active_missile(unit)

        assert result is True

    def test_is_active_missile_unit_type(self):
        """Test _is_active_missile with unit_type='missile'."""
        unit = MockUnit(is_missile=False, unit_type="missile")

        result = MissileCCDManager._is_active_missile(unit)

        assert result is True

    def test_is_active_missile_class_name(self):
        """Test _is_active_missile with class name ending in 'missile'."""
        unit = MockUnit(is_missile=False, class_name="AIM120Missile")

        result = MissileCCDManager._is_active_missile(unit)

        assert result is True

    def test_is_active_missile_false(self):
        """Test _is_active_missile returns False for non-missiles."""
        unit = MockUnit(is_missile=False, unit_type="aircraft", class_name="F16")

        result = MissileCCDManager._is_active_missile(unit)

        assert result is False


class TestMissileCCDManagerPostUpdateCCD:
    """Test post_update_ccd method."""

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_no_active_missiles(self, mock_distance):
        """Test with no active missiles."""
        manager = MissileCCDManager()
        units = [MockUnit(is_missile=False), MockUnit(is_missile=False)]
        sim = Mock()

        manager.post_update_ccd(units, 1.0, sim)

        mock_distance.assert_not_called()
        sim.log_event.assert_not_called()

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_missile_without_target(self, mock_distance):
        """Test missile without target."""
        manager = MissileCCDManager()
        manager.target_selector = Mock(return_value=None)

        missile = MockUnit(is_missile=True)
        units = [missile]
        sim = Mock()

        manager.post_update_ccd(units, 1.0, sim)

        mock_distance.assert_not_called()
        sim.log_event.assert_not_called()

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_missile_with_destroyed_target(self, mock_distance):
        """Test missile with destroyed target."""
        manager = MissileCCDManager()
        target = MockUnit(is_destroyed=True)
        manager.target_selector = Mock(return_value=target)

        missile = MockUnit(is_missile=True)
        units = [missile, target]
        sim = Mock()

        manager.post_update_ccd(units, 1.0, sim)

        mock_distance.assert_not_called()
        sim.log_event.assert_not_called()

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_target_beyond_max_range(self, mock_distance):
        """Test target beyond max consideration range."""
        mock_distance.return_value = 60.0  # 60 km, beyond default 50 km max

        manager = MissileCCDManager()
        target = MockUnit()
        manager.target_selector = Mock(return_value=target)

        missile = MockUnit(is_missile=True)
        units = [missile, target]
        sim = Mock()

        manager.post_update_ccd(units, 1.0, sim)

        mock_distance.assert_called_once()
        sim.log_event.assert_not_called()  # Should not engage

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_target_outside_lock_gate_no_ccd(self, mock_distance):
        """Test target outside lock gate - no CCD performed."""
        mock_distance.return_value = 15.0  # 15 km, outside default 10 km gate

        manager = MissileCCDManager()
        target = MockUnit()
        manager.target_selector = Mock(return_value=target)

        missile = MockUnit(is_missile=True)
        units = [missile, target]
        sim = Mock()

        with patch.object(manager.hit_calc, "check_and_maybe_hit") as mock_hit_check:
            manager.post_update_ccd(units, 1.0, sim)

            mock_hit_check.assert_not_called()  # Should not perform CCD
            sim.log_event.assert_not_called()  # Should not log engagement

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_first_time_within_lock_gate_logs_engagement(self, mock_distance):
        """Test first time within lock gate logs engagement event."""
        mock_distance.return_value = 8.0  # Within default 10 km gate

        manager = MissileCCDManager()
        target = MockUnit()
        manager.target_selector = Mock(return_value=target)

        missile = MockUnit(unit_id=1, is_missile=True)
        units = [missile, target]
        sim = Mock()

        with patch.object(manager.hit_calc, "check_and_maybe_hit") as mock_hit_check:
            with patch(
                "air_to_air_rl.simulator.core.hit_event_helpers.MissileEngagementEvent"
            ) as mock_event:
                mock_hit_check.return_value = (False, None, None)  # No hit
                mock_event.return_value = "engagement_event"

                manager.post_update_ccd(units, 1.0, sim)

                # Should log engagement event
                mock_event.assert_called_once_with(sim, missile, target)
                sim.log_event.assert_called_once_with("engagement_event")

                # Should track that this missile has engaged
                assert 1 in manager._engaged_once

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_repeated_within_lock_gate_no_duplicate_engagement(self, mock_distance):
        """Test repeated entries within lock gate don't log duplicate engagements."""
        mock_distance.return_value = 8.0

        manager = MissileCCDManager()
        manager._engaged_once.add(1)  # Already engaged
        target = MockUnit()
        manager.target_selector = Mock(return_value=target)

        missile = MockUnit(unit_id=1, is_missile=True)
        units = [missile, target]
        sim = Mock()

        with patch.object(manager.hit_calc, "check_and_maybe_hit") as mock_hit_check:
            mock_hit_check.return_value = (False, None, None)

            manager.post_update_ccd(units, 1.0, sim)

            # Should not log engagement event
            sim.log_event.assert_not_called()

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_successful_hit_calls_on_hit(self, mock_distance):
        """Test successful hit calls on_hit callback."""
        mock_distance.return_value = 5.0  # Within lock gate

        manager = MissileCCDManager()
        manager.on_hit = Mock()
        target = MockUnit()
        manager.target_selector = Mock(return_value=target)

        missile = MockUnit(unit_id=1, is_missile=True)
        units = [missile, target]
        sim = Mock()

        with patch.object(manager.hit_calc, "check_and_maybe_hit") as mock_hit_check:
            mock_hit_check.return_value = (True, target, 0.75)  # Hit!

            manager.post_update_ccd(units, 1.0, sim)

            manager.on_hit.assert_called_once_with(missile, target, 0.75, sim)

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_hit_with_none_t_frac_uses_zero(self, mock_distance):
        """Test hit with None t_frac uses 0.0."""
        mock_distance.return_value = 5.0

        manager = MissileCCDManager()
        manager.on_hit = Mock()
        target = MockUnit()
        manager.target_selector = Mock(return_value=target)

        missile = MockUnit(unit_id=1, is_missile=True)
        units = [missile, target]
        sim = Mock()

        with patch.object(manager.hit_calc, "check_and_maybe_hit") as mock_hit_check:
            mock_hit_check.return_value = (True, target, None)  # t_frac is None

            manager.post_update_ccd(units, 1.0, sim)

            manager.on_hit.assert_called_once_with(missile, target, 0.0, sim)

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_multiple_missiles_processed(self, mock_distance):
        """Test multiple missiles are all processed."""
        mock_distance.return_value = 5.0  # All within lock gate

        manager = MissileCCDManager()
        target1 = MockUnit(unit_id=10)
        target2 = MockUnit(unit_id=20)
        manager.target_selector = Mock(side_effect=[target1, target2])

        missile1 = MockUnit(unit_id=1, is_missile=True)
        missile2 = MockUnit(unit_id=2, is_missile=True)
        units = [missile1, missile2, target1, target2]
        sim = Mock()

        with patch.object(manager.hit_calc, "check_and_maybe_hit") as mock_hit_check:
            mock_hit_check.return_value = (False, None, None)

            manager.post_update_ccd(units, 1.0, sim)

            # Should check both missiles
            assert mock_hit_check.call_count == 2
            assert mock_distance.call_count == 2

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_units_list_copy_prevents_iteration_error(self, mock_distance):
        """Test that units are copied to prevent iteration errors."""
        mock_distance.return_value = 5.0

        manager = MissileCCDManager()
        manager.on_hit = Mock()
        target = MockUnit()
        manager.target_selector = Mock(return_value=target)

        missile = MockUnit(unit_id=1, is_missile=True)

        # Mock units that change during iteration
        units_mock = Mock()
        units_mock.__iter__ = Mock(return_value=iter([missile, target]))

        sim = Mock()

        with patch.object(manager.hit_calc, "check_and_maybe_hit") as mock_hit_check:
            mock_hit_check.return_value = (False, None, None)

            # Should not raise iteration error
            manager.post_update_ccd(units_mock, 1.0, sim)


class TestMissileCCDManagerMaintenance:
    """Test maintenance methods of MissileCCDManager."""

    def test_reset_engagement_tracking(self):
        """Test resetting engagement tracking."""
        manager = MissileCCDManager()
        manager._engaged_once.add(1)
        manager._engaged_once.add(2)
        manager._engaged_once.add(3)

        manager.reset_engagement_tracking()

        assert len(manager._engaged_once) == 0

    def test_reset_engagement_tracking_empty(self):
        """Test resetting empty engagement tracking."""
        manager = MissileCCDManager()

        # Should not raise error
        manager.reset_engagement_tracking()

        assert len(manager._engaged_once) == 0


class TestMissileCCDManagerWithCountermeasures:
    """Test MissileCCDManager with countermeasures present."""

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_countermeasure_not_processed_as_missile(self, mock_distance):
        """Test that countermeasures are not processed as missiles."""
        mock_distance.return_value = 5.0

        manager = MissileCCDManager()

        # Create countermeasure (not a missile)
        countermeasure = MockUnit(unit_id=1, is_missile=False)
        countermeasure.is_countermeasure = True

        target = MockUnit(unit_id=2)
        sim = Mock()

        with patch.object(manager.hit_calc, "check_and_maybe_hit") as mock_hit_check:
            manager.post_update_ccd([countermeasure, target], 1.0, sim)

            # Should not perform CCD for countermeasure
            mock_hit_check.assert_not_called()
            sim.log_event.assert_not_called()

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_missile_targeting_countermeasure_scenario(self, mock_distance):
        """Test scenario where missile's target selector might return countermeasure."""
        mock_distance.return_value = 5.0

        manager = MissileCCDManager()

        # Create missile
        missile = MockUnit(unit_id=1, is_missile=True)

        # Create countermeasure
        countermeasure = MockUnit(unit_id=2)
        countermeasure.is_countermeasure = True

        # Target selector returns countermeasure (edge case)
        manager.target_selector = Mock(return_value=countermeasure)

        sim = Mock()

        with patch.object(manager.hit_calc, "check_and_maybe_hit") as mock_hit_check:
            mock_hit_check.return_value = (False, None, None)

            manager.post_update_ccd([missile, countermeasure], 1.0, sim)

            # CCD should still be performed, but countermeasure can be filtered
            # by custom logic if needed
            mock_hit_check.assert_called_once()

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_mixed_units_with_countermeasures(self, mock_distance):
        """Test CCD with mix of missiles, targets, and countermeasures."""
        mock_distance.return_value = 5.0

        manager = MissileCCDManager()

        # Create various unit types
        missile = MockUnit(unit_id=1, is_missile=True)
        target = MockUnit(unit_id=2)
        countermeasure1 = MockUnit(unit_id=3, is_missile=False)
        countermeasure1.is_countermeasure = True
        countermeasure2 = MockUnit(unit_id=4, is_missile=False)
        countermeasure2.is_countermeasure = True

        # Missile targets the actual target, not countermeasures
        missile.target = target
        manager.target_selector = Mock(return_value=target)

        sim = Mock()

        with patch.object(manager.hit_calc, "check_and_maybe_hit") as mock_hit_check:
            mock_hit_check.return_value = (False, None, None)

            units = [missile, target, countermeasure1, countermeasure2]
            manager.post_update_ccd(units, 1.0, sim)

            # Should only process missile, not countermeasures
            assert mock_hit_check.call_count == 1  # Only for the missile


class TestMissileCCDManagerIntegration:
    """Integration tests for complete CCD scenarios."""

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_complete_engagement_scenario(self, mock_distance):
        """Test complete engagement from first detection to hit."""
        mock_distance.return_value = 8.0  # Within lock gate

        config = CCDConfig(within_lock_gate_m=10000.0, fuse_radius_m=300.0)
        manager = MissileCCDManager(config)

        target = MockUnit(unit_id=2)
        missile = MockUnit(unit_id=1, is_missile=True, target=target)
        sim = Mock()

        # First call - engagement logged, no hit
        with patch.object(manager.hit_calc, "check_and_maybe_hit") as mock_hit_check:
            mock_hit_check.return_value = (False, None, None)

            manager.post_update_ccd([missile, target], 1.0, sim)

            # Should log engagement
            assert sim.log_event.call_count == 1
            assert 1 in manager._engaged_once

        # Reset sim mock for second call
        sim.log_event.reset_mock()

        # Second call - hit occurs
        with patch.object(manager.hit_calc, "check_and_maybe_hit") as mock_hit_check:
            mock_hit_check.return_value = (True, target, 0.5)
            manager.on_hit = Mock()

            manager.post_update_ccd([missile, target], 1.0, sim)

            # Should not log another engagement, but should hit
            sim.log_event.assert_not_called()
            manager.on_hit.assert_called_once_with(missile, target, 0.5, sim)

    @patch("air_to_air_rl.simulator.core.hit_event_helpers.units_distance_km")
    def test_multiple_missiles_same_target(self, mock_distance):
        """Test multiple missiles engaging same target."""
        mock_distance.return_value = 6.0  # All within range

        manager = MissileCCDManager()
        target = MockUnit(unit_id=10)
        missile1 = MockUnit(unit_id=1, is_missile=True, target=target)
        missile2 = MockUnit(unit_id=2, is_missile=True, target=target)

        sim = Mock()

        with patch.object(manager.hit_calc, "check_and_maybe_hit") as mock_hit_check:
            # First missile hits, second misses
            mock_hit_check.side_effect = [(True, target, 0.3), (False, None, None)]
            manager.on_hit = Mock()

            manager.post_update_ccd([missile1, missile2, target], 1.0, sim)

            # Should process both missiles
            assert mock_hit_check.call_count == 2
            # Should hit with first missile
            manager.on_hit.assert_called_once_with(missile1, target, 0.3, sim)
            # Should log two engagement events
            assert sim.log_event.call_count == 2

    def test_state_capture_and_restore_integration(self):
        """Test integration of state capture and restore functionality."""
        manager = MissileCCDManager()

        # Create unit with complex state
        unit = MockUnit(unit_id=1)
        unit.yaw_deg = 45.0
        unit.pitch_deg = 10.0
        unit.speed = 300.0
        unit.desired_yaw_deg = 50.0
        unit.position.lat = 47.0
        unit.position.lon = -120.0
        unit.position.alt = 5000.0

        # Take snapshot
        manager.begin_tick_snapshot([unit])

        # Modify unit state
        unit.yaw_deg = 90.0
        unit.pitch_deg = -5.0
        unit.speed = 250.0
        unit.position.lat = 48.0

        # Get and apply begin state
        begin_state = manager.get_begin_state(unit)
        assert begin_state is not None

        manager.apply_state(unit, begin_state)

        # Should be restored to original state
        assert unit.yaw_deg == 45.0
        assert unit.pitch_deg == 10.0
        assert unit.speed == 300.0
        assert unit.desired_yaw_deg == 50.0
        # Position should also be restored
        assert unit.position.lat == 47.0
        assert unit.position.lon == -120.0
        assert unit.position.alt == 5000.0
