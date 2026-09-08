#!/usr/bin/env python3
"""
Sanity tests for weapon firing logic:
(a) gun fires independently
(b) missile fires only with lock+FOV+inventory+cooldown
(c) cooldown prevents repeated shots
"""

import logging
from unittest.mock import MagicMock, Mock

import numpy as np

from tests.mocks.identity import declare_unit_identity


class MockPosition:
    def __init__(self):
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 10000.0

    def copy(self):
        pos = MockPosition()
        pos.lat = self.lat
        pos.lon = self.lon
        pos.alt = self.alt
        return pos


def setup_mock_unit_and_simulator():
    """Create mock unit and simulator for testing."""

    # Mock simulator
    simulator = Mock()
    simulator.active_units = {}
    simulator.tick_secs = 0.1
    simulator.count_missiles_at_target = Mock(return_value=0)

    # Mock unit
    unit = Mock()
    unit.id = "test_aircraft"
    unit.group = "agent"
    unit.speed = 250.0
    unit.min_speed_mps = 50.0  # Add min_speed_mps
    unit.pitch_deg = 0.0
    unit.yaw_deg = 0.0
    unit.max_speed_mps = 400.0
    unit.missile_types = ["test_missile_class"]
    unit.remaining_missiles = 4
    # Real identity, not fabricated: the action processor walks unit.missiles.
    declare_unit_identity(unit)

    # Mock position
    unit.position = MockPosition()

    # Mock control
    unit.control = Mock()
    unit.control.throttle = 0.5
    unit.control.set_throttle = Mock()
    unit.control.set_yaw_deg = Mock()
    unit.control.set_pitch_deg = Mock()

    # Mock physics
    unit.physics = Mock()
    unit.physics.g = 9.81
    unit.physics.n_max = 9.0
    unit.physics.mass_kg = 15000.0
    unit.physics.A_m2 = 50.0
    unit.physics.max_pitch_rate_deg_s = 20.0  # Add max_pitch_rate_deg_s
    unit.physics.max_turn_rate_deg_s = 30.0  # Add max_turn_rate_deg_s
    unit.physics.service_ceiling = 15000.0  # Add service_ceiling
    unit.physics.air = Mock()
    unit.physics.air.get_density = Mock(return_value=1.0)
    unit.physics.get_base_drag_cd = Mock(return_value=0.02)
    unit.physics.get_engine_force = Mock(return_value=100000.0)
    unit.physics.compute_total_drag = Mock(return_value=10000.0)  # Add to fix drag calculation
    unit.physics.compute_instantaneous_turn_rate = Mock(return_value=15.0)  # Add turn rate
    unit.physics.get_pitch_limit_deg = Mock(return_value=60.0)  # Add pitch limit
    unit.physics._dynamic_stall = Mock(return_value=50.0)  # Add dynamic stall

    # Mock sensor
    unit.sensor = Mock()
    unit.sensor.has_radar_lock = Mock(return_value=True)

    # Mock weapons
    unit.weapons = Mock()
    unit.weapons.is_target_in_fov = Mock(return_value=True)
    unit.weapons.fire_missile = Mock()
    unit.weapons.fire_gun = Mock()

    # Mock countermeasures
    unit.countermeasures = Mock()
    unit.countermeasures.launch_flares = Mock()
    unit.countermeasures.launch_chaff = Mock()
    unit.countermeasures.activate_ecm = Mock()
    unit.countermeasures.deploy_decoys = Mock()

    return simulator, unit


def test_gun_fires_independently():
    """Test (a): gun fires independently of missile logic."""
    print("=== Test (a): Gun fires independently ===")

    from bvr_marl_core.rl.environment.spaces.action_space import ActionProcessor

    simulator, unit = setup_mock_unit_and_simulator()
    processor = ActionProcessor(simulator, information_mode="oracle")

    # Add unit to simulator
    simulator.active_units[1] = unit

    # Mock target
    target = Mock()
    target.id = "target_1"
    target.group = "enemy"  # Add group attribute
    target.is_missile = False  # Add is_missile attribute
    target.is_non_engageable = False  # Must be False or Mock() is truthy and filters the target
    target.position = MockPosition()
    target.position.lat = 0.1
    target.position.lon = 0.1
    target.position.alt = 10000.0

    # Add target to simulator so it can be found
    simulator.active_units["target_1"] = target

    # Set up gun to return successful fire
    unit.weapons.fire_gun.return_value = ["projectile1", "projectile2"]

    # Action: no missile fire (0.0), gun fire (1.0)
    action = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0])

    # Apply action
    processor.apply(1, action)

    # Check that gun was fired but missile was not
    unit.weapons.fire_gun.assert_called_once()
    unit.weapons.fire_missile.assert_not_called()

    # Check training signals
    signals = processor.get_training_signals(1)
    assert signals["valid_gun_fires"] == 1
    assert signals["valid_missile_fires"] == 0

    print("OK Gun fires independently of missile logic")


def test_missile_requires_all_conditions():
    """Test (b): missile fires only with lock+FOV+inventory+cooldown."""
    print("\n=== Test (b): Missile requires all conditions ===")

    from bvr_marl_core.rl.environment.spaces.action_space import ActionProcessor

    simulator, unit = setup_mock_unit_and_simulator()
    processor = ActionProcessor(simulator, information_mode="oracle")

    # Add unit to simulator
    simulator.active_units[1] = unit

    # Mock target
    target = Mock()
    target.id = "target_1"
    target.group = "enemy"  # Add group attribute
    target.is_missile = False  # Add is_missile attribute
    target.is_non_engageable = False  # Must be False or Mock() is truthy and filters the target
    target.position = MockPosition()
    target.position.lat = 0.1
    target.position.lon = 0.1
    target.position.alt = 10000.0

    # Add target to simulator so it can be found
    simulator.active_units["target_1"] = target

    # Set up missile to return successful fire
    # fire_missile returns (missile, veto_reason, diagnostics)
    mock_missile = Mock()
    unit.weapons.fire_missile.return_value = (mock_missile, None, {})

    # Test case 1: All conditions met - should fire
    unit.sensor.has_radar_lock.return_value = True
    unit.weapons.is_target_in_fov.return_value = True
    unit.remaining_missiles = 4

    action = np.array([0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    processor.apply(1, action)

    signals = processor.get_training_signals(1)
    assert signals["valid_missile_fires"] == 1, "Should fire when all conditions met"

    # Reset for next test
    unit.weapons.fire_missile.reset_mock()

    # Test case 2: No radar lock - should not fire
    unit.sensor.has_radar_lock.return_value = False
    processor.apply(1, action)

    signals = processor.get_training_signals(1)
    assert signals["vetoed_missile_attempts"] == 1, "Should veto when no radar lock"

    # Reset for next test
    unit.weapons.fire_missile.reset_mock()
    unit.sensor.has_radar_lock.return_value = True

    # Test case 3: Not in FOV - should not fire
    unit.weapons.is_target_in_fov.return_value = False
    processor.apply(1, action)

    signals = processor.get_training_signals(1)
    assert signals["vetoed_missile_attempts"] == 1, "Should veto when not in FOV"

    # Reset for next test
    unit.weapons.fire_missile.reset_mock()
    unit.weapons.is_target_in_fov.return_value = True

    # Test case 4: No missiles remaining - should not fire
    unit.remaining_missiles = 0
    processor.apply(1, action)

    signals = processor.get_training_signals(1)
    assert signals["vetoed_missile_attempts"] == 1, "Should veto when no missiles"

    print("OK Missile requires lock+FOV+inventory conditions")


def test_cooldown_prevents_repeated_shots():
    """Test (c): cooldown prevents repeated shots."""
    print("\n=== Test (c): Cooldown prevents repeated shots ===")

    from bvr_marl_core.rl.environment.spaces.action_space import ActionProcessor

    simulator, unit = setup_mock_unit_and_simulator()
    processor = ActionProcessor(simulator, information_mode="oracle")

    # Add unit to simulator
    simulator.active_units[1] = unit

    # Mock target
    target = Mock()
    target.id = "target_1"
    target.group = "enemy"  # Add group attribute
    target.is_missile = False  # Add is_missile attribute
    target.is_non_engageable = False  # Must be False or Mock() is truthy and filters the target
    target.position = MockPosition()
    target.position.lat = 0.1
    target.position.lon = 0.1
    target.position.alt = 10000.0

    # Add target to simulator so it can be found
    simulator.active_units["target_1"] = target

    # Set up all conditions for successful firing
    unit.sensor.has_radar_lock.return_value = True
    unit.weapons.is_target_in_fov.return_value = True
    unit.remaining_missiles = 4
    # fire_missile returns (missile, veto_reason, diagnostics)
    mock_missile = Mock()
    unit.weapons.fire_missile.return_value = (mock_missile, None, {})
    unit.weapons.fire_gun.return_value = ["projectile"]

    # Action with both missile and gun fire
    action = np.array([0.5, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0])

    # First shot - should work
    processor.apply(1, action)
    signals1 = processor.get_training_signals(1)

    assert signals1["valid_missile_fires"] == 1, "First missile shot should succeed"
    assert signals1["valid_gun_fires"] == 1, "First gun shot should succeed"

    # Check cooldown state
    state = processor.agent_states[1]
    assert state["missile_cooldown_left_s"] > 0, "Missile cooldown should be active"
    assert state["gun_cooldown_left_s"] > 0, "Gun cooldown should be active"

    # Reset mocks for second shot
    unit.weapons.fire_missile.reset_mock()
    unit.weapons.fire_gun.reset_mock()

    # Second shot immediately - should be vetoed due to cooldown
    processor.apply(1, action)
    signals2 = processor.get_training_signals(1)

    assert signals2["vetoed_missile_attempts"] == 1, (
        "Second missile shot should be vetoed by cooldown"
    )
    assert signals2["vetoed_gun_attempts"] == 1, "Second gun shot should be vetoed by cooldown"

    # Simulate time passing (cooldown expires)
    state["missile_cooldown_left_s"] = 0.0
    state["gun_cooldown_left_s"] = 0.0

    # Third shot after cooldown - should work again
    processor.apply(1, action)
    signals3 = processor.get_training_signals(1)

    assert signals3["valid_missile_fires"] == 1, "Third missile shot should succeed after cooldown"
    assert signals3["valid_gun_fires"] == 1, "Third gun shot should succeed after cooldown"

    print("OK Cooldown prevents repeated shots correctly")


def test_envelope_scalars_exposure():
    """Test that envelope scalars are properly exposed."""
    print("\n=== Test: Envelope scalars exposure ===")

    from bvr_marl_core.rl.environment.spaces.action_space import ActionProcessor

    simulator, unit = setup_mock_unit_and_simulator()
    processor = ActionProcessor(simulator, information_mode="oracle")

    # Add unit to simulator
    simulator.active_units[1] = unit

    # Initialize agent state
    processor._init_agent_state(1, unit)

    # Get envelope scalars
    scalars = processor.get_envelope_scalars(1)

    # Check that required scalars are present
    assert "s_factor" in scalars, "s_factor should be present"
    assert "c_factor" in scalars, "c_factor should be present"
    assert isinstance(scalars["s_factor"], (int, float)), "s_factor should be numeric"
    assert isinstance(scalars["c_factor"], (int, float)), "c_factor should be numeric"

    print("OK Envelope scalars properly exposed")


if __name__ == "__main__":
    # Disable logging for cleaner test output
    logging.disable(logging.CRITICAL)

    print("Running weapon firing sanity tests...\n")

    tests = [
        test_gun_fires_independently,
        test_missile_requires_all_conditions,
        test_cooldown_prevents_repeated_shots,
        test_envelope_scalars_exposure,
    ]

    results = []
    for test in tests:
        try:
            test()  # Tests now use assertions instead of returning values
            results.append(True)
            print(f"PASS {test.__name__}")
        except Exception as e:
            print(f"FAIL Test {test.__name__} crashed: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)

    print("\n=== Summary ===")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("SUCCESS All weapon firing sanity tests passed!")
        exit(0)
    else:
        print("FAILURE Some tests failed - please review the implementation")
        exit(1)
