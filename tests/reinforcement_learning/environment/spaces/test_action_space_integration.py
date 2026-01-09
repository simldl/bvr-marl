"""
Integration tests for the enhanced action space system.

Tests the TODO integration items:
- Gun fires independently from missile
- Missile fires only with lock & FOV
- Cooldown works
- Target stickiness is time-based
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from reinforcement_learning.environment.spaces.action_space import ActionProcessor


class TestActionSpaceIntegration:

    @pytest.fixture
    def mock_simulator(self):
        sim = Mock()
        sim.tick_secs = 0.1
        sim.active_units = {}
        return sim

    @pytest.fixture
    def mock_unit(self):
        unit = Mock()
        unit.id = "test_unit"
        unit.group = "BLUE"  # Add group attribute
        unit.speed = 250.0
        unit.min_speed_mps = 50.0
        unit.max_speed_mps = 300.0
        unit.pitch_deg = 0.0
        unit.yaw_deg = 0.0
        unit.missile_types = ["AIM-120"]
        unit.remaining_missiles = 4  # Add missile inventory

        # Mock position
        unit.position = Mock()
        unit.position.lat = 35.0
        unit.position.lon = -115.0
        unit.position.alt = 10000.0

        # Mock control
        unit.control = Mock()
        unit.control.throttle = 0.5
        unit.control.set_yaw_deg = Mock()
        unit.control.set_pitch_deg = Mock()
        unit.control.set_throttle = Mock()

        # Mock physics
        unit.physics = Mock()
        unit.physics.g = 9.81
        unit.physics.mass_kg = 15000.0
        unit.physics.A_m2 = 25.0
        unit.physics.n_max = 9.0
        unit.physics.max_pitch_rate_deg_s = 20.0
        unit.physics.max_turn_rate_deg_s = 30.0
        unit.physics.service_ceiling = 15000.0
        unit.physics.air = Mock()
        unit.physics.air.get_density = Mock(return_value=0.8)
        unit.physics.get_base_drag_cd = Mock(return_value=0.02)
        unit.physics.get_engine_force = Mock(return_value=50000.0)
        unit.physics.compute_total_drag = Mock(return_value=10000.0)  # Add this to fix the drag calculation
        unit.physics.compute_instantaneous_turn_rate = Mock(return_value=15.0)
        unit.physics.get_pitch_limit_deg = Mock(return_value=60.0)  # Add pitch limit
        unit.physics._dynamic_stall = Mock(return_value=50.0)

        # Mock sensor and weapons systems
        unit.sensor = Mock()
        unit.sensor.has_radar_lock = Mock(return_value=True)

        unit.weapons = Mock()
        unit.weapons.is_target_in_fov = Mock(return_value=True)
        # fire_missile will be configured in each test based on conditions
        # Mimics the real fire_missile gating logic
        mock_missile = Mock()
        def fire_missile_side_effect(sim, target, missile_cls):
            # Check radar lock (mimics real weapon system)
            if not unit.sensor.has_radar_lock(target):
                return (None, "no_radar_lock", {})
            # Check FOV (mimics real weapon system)
            if not unit.weapons.is_target_in_fov(target):
                return (None, "not_in_fov", {})
            # Success
            return (mock_missile, None, {})
        unit.weapons.fire_missile = Mock(side_effect=fire_missile_side_effect)

        def fire_gun_side_effect(sim, target):
            # Check FOV for gun
            if not unit.weapons.is_target_in_fov(target):
                return []
            return [Mock()]
        unit.weapons.fire_gun = Mock(side_effect=fire_gun_side_effect)

        # Add countermeasures mock
        unit.countermeasures = Mock()
        unit.countermeasures.launch_flares = Mock()
        unit.countermeasures.launch_chaff = Mock()
        unit.countermeasures.activate_ecm = Mock()
        unit.countermeasures.deploy_decoys = Mock()

        return unit

    @pytest.fixture
    def mock_target(self):
        target = Mock()
        target.id = "enemy_1"
        target.group = "RED"  # Add group attribute
        target.is_missile = False  # Not a missile
        target.position = Mock()
        target.position.lat = 35.1
        target.position.lon = -115.1
        target.position.alt = 10000.0
        return target

    @pytest.fixture
    def action_processor(self, mock_simulator):
        return ActionProcessor(mock_simulator)

    def test_missile_fires_only_with_lock_and_fov(self, action_processor, mock_simulator, mock_unit, mock_target):
        """Test that missile fires only when both lock_ok=True and fov_ok=True"""
        # Setup
        agent_id = 1
        mock_simulator.active_units[agent_id] = mock_unit

        action = np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0])  # action[4]=0.8

        # Track fire results
        fire_results = []
        original_fire = mock_unit.weapons.fire_missile.side_effect
        def track_fire(*args, **kwargs):
            result = original_fire(*args, **kwargs)
            fire_results.append(result)
            return result
        mock_unit.weapons.fire_missile.side_effect = track_fire

        # Test 1: With lock and FOV - should fire successfully
        mock_simulator.active_units[mock_target.id] = mock_target
        mock_unit.sensor.has_radar_lock.return_value = True
        mock_unit.weapons.is_target_in_fov.return_value = True

        with patch.object(action_processor.target_sorter, 'sort_target_candidates', return_value=[mock_target]):
            action_processor.apply(agent_id, action)

        # Check that fire_missile was called and returned a missile (not vetoed)
        assert len(fire_results) == 1
        missile, veto, _ = fire_results[0]
        assert missile is not None, "Missile should have been created with lock and FOV"
        assert veto is None
        fire_results.clear()

        # Test 2: Without lock - fire_missile should be called but return veto
        mock_unit.sensor.has_radar_lock.return_value = False
        mock_unit.weapons.is_target_in_fov.return_value = True

        # Reset agent state for clean test
        action_processor.reset_agent_state(agent_id)

        with patch.object(action_processor.target_sorter, 'sort_target_candidates', return_value=[mock_target]):
            action_processor.apply(agent_id, action)

        # fire_missile is called, but should return veto (None missile)
        assert len(fire_results) == 1
        missile, veto, _ = fire_results[0]
        assert missile is None, "Missile should be vetoed without lock"
        assert veto == "no_radar_lock", "Veto reason should be no_radar_lock"
        fire_results.clear()

        # Test 3: Without FOV - fire_missile should be called but return veto
        mock_unit.sensor.has_radar_lock.return_value = True
        mock_unit.weapons.is_target_in_fov.return_value = False

        # Reset agent state for clean test
        action_processor.reset_agent_state(agent_id)

        with patch.object(action_processor.target_sorter, 'sort_target_candidates', return_value=[mock_target]):
            action_processor.apply(agent_id, action)

        # fire_missile is called, but should return veto (None missile)
        assert len(fire_results) == 1
        missile, veto, _ = fire_results[0]
        assert missile is None, "Missile should be vetoed without FOV"
        assert veto == "not_in_fov", "Veto reason should be not_in_fov"

    def test_cooldown_works(self, action_processor, mock_simulator, mock_unit, mock_target):
        """Test that cooldown prevents repeated firing"""
        # Setup
        agent_id = 1
        mock_simulator.active_units[agent_id] = mock_unit
        mock_simulator.tick_secs = 0.1  # 100ms time step

        action = np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0])  # action[4]=0.8

        # Mock successful conditions
        mock_simulator.active_units[mock_target.id] = mock_target
        mock_unit.sensor.has_radar_lock.return_value = True
        mock_unit.weapons.is_target_in_fov.return_value = True

        with patch.object(action_processor.target_sorter, 'sort_target_candidates', return_value=[mock_target]):
            # First shot should work
            action_processor.apply(agent_id, action)
            assert mock_unit.weapons.fire_missile.called
            mock_unit.weapons.fire_missile.reset_mock()

            # Immediate second shot should be blocked by cooldown
            action_processor.apply(agent_id, action)
            mock_unit.weapons.fire_missile.assert_not_called()

            # After cooldown period (2.0s), should work again
            # Simulate enough time steps to exceed cooldown
            for _ in range(25):  # 25 * 0.1s = 2.5s > 2.0s cooldown
                action_processor.apply(agent_id, np.zeros(10))  # Dummy actions to advance time

            # Now missile should fire again
            action_processor.apply(agent_id, action)
            assert mock_unit.weapons.fire_missile.called

    def test_target_stickiness_is_time_based(self, action_processor, mock_simulator, mock_unit):
        """Test that target hold behavior is time-based, not step-based"""
        # Setup
        agent_id = 1
        mock_simulator.active_units[agent_id] = mock_unit

        # Create two targets at different ranges
        target1 = Mock()
        target1.id = "enemy_1"
        target1.group = "RED"
        target1.is_missile = False
        target1.position = Mock()
        target1.position.lat = 35.05  # Closer
        target1.position.lon = -115.05
        target1.position.alt = 10000.0

        target2 = Mock()
        target2.id = "enemy_2"
        target2.group = "RED"
        target2.is_missile = False
        target2.position = Mock()
        target2.position.lat = 35.1   # Farther
        target2.position.lon = -115.1
        target2.position.alt = 10000.0

        targets = [target1, target2]

        # First action selects target 0 (closer)
        action1 = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # action[3]=0.0
        with patch.object(action_processor.target_sorter, 'sort_target_candidates', return_value=targets):
            action_processor.apply(agent_id, action1)

        state = action_processor.agent_states[agent_id]
        assert state['target_index'] == 0  # Should select first target

        # Try to change to target 1 immediately - should be sticky
        action2 = np.array([0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # action[3]=1.0
        with patch.object(action_processor.target_sorter, 'sort_target_candidates', return_value=targets):
            action_processor.apply(agent_id, action2)

        # Should still be target 0 due to stickiness
        assert state['target_index'] == 0

        # Test different timesteps with same total time
        action_processor.reset_agent_state(agent_id)

        # Test 1: Large timestep
        mock_simulator.tick_secs = 2.5  # > tau_hold_s (2.0)
        with patch.object(action_processor.target_sorter, 'sort_target_candidates', return_value=targets):
            action_processor.apply(agent_id, action1)  # Select target 0
            action_processor.apply(agent_id, action2)  # Try to select target 1

        state1 = action_processor.agent_states[agent_id]
        target_after_large_dt = state1['target_index']

        # Test 2: Small timesteps with same total time
        action_processor.reset_agent_state(agent_id)
        mock_simulator.tick_secs = 0.1  # Small timestep

        with patch.object(action_processor.target_sorter, 'sort_target_candidates', return_value=targets):
            action_processor.apply(agent_id, action1)  # Select target 0
            # Apply many small steps totaling > 2.0s
            for _ in range(25):  # 25 * 0.1 = 2.5s
                action_processor.apply(agent_id, np.zeros(10))
            action_processor.apply(agent_id, action2)  # Try to select target 1

        state2 = action_processor.agent_states[agent_id]
        target_after_small_dt = state2['target_index']

        # Both should behave the same (time-based, not step-based)
        assert target_after_large_dt == target_after_small_dt

    def test_training_signals(self, action_processor, mock_simulator, mock_unit, mock_target):
        """Test that training signals are properly tracked for reward hygiene"""
        # Setup
        agent_id = 1
        mock_simulator.active_units[agent_id] = mock_unit

        action = np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0])

        # Test valid fire
        mock_simulator.active_units[mock_target.id] = mock_target
        mock_unit.sensor.has_radar_lock.return_value = True
        mock_unit.weapons.is_target_in_fov.return_value = True

        with patch.object(action_processor.target_sorter, 'sort_target_candidates', return_value=[mock_target]):
            action_processor.apply(agent_id, action)

        signals = action_processor.get_training_signals(agent_id)
        assert signals['valid_missile_fires'] == 1
        assert signals['valid_gun_fires'] == 1
        assert signals['vetoed_missile_attempts'] == 0
        assert signals['vetoed_gun_attempts'] == 0

        # Test vetoed fire
        action_processor.reset_agent_state(agent_id)
        mock_unit.sensor.has_radar_lock.return_value = False  # Will veto missile
        mock_unit.weapons.is_target_in_fov.return_value = False  # Will veto gun

        with patch.object(action_processor.target_sorter, 'sort_target_candidates', return_value=[mock_target]):
            action_processor.apply(agent_id, action)

        signals = action_processor.get_training_signals(agent_id)
        assert signals['valid_missile_fires'] == 0
        assert signals['valid_gun_fires'] == 0
        assert signals['vetoed_missile_attempts'] == 1
        assert signals['vetoed_gun_attempts'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])