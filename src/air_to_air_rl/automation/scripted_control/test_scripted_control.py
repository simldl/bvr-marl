"""
Test suite for the fully scripted tactical controller.
Validates advanced air-to-air combat tactics and behavior tree execution.
"""

import sys

import numpy as np

from air_to_air_rl.automation.scripted_control.behavior_tree.nodes import (
    ConditionNode,
    NodeStatus,
    SelectorNode,
)
from air_to_air_rl.automation.scripted_control.full_tactical_controller import (
    FullTacticalController,
)
from air_to_air_rl.automation.scripted_control.tactics.bvr_tactics import BVRTactics
from air_to_air_rl.automation.scripted_control.tactics.energy_management import EnergyManager
from air_to_air_rl.automation.scripted_control.tactics.geometry_calculator import GeometryCalculator
from air_to_air_rl.automation.scripted_control.tactics.range_calculator import RangeCalculator
from air_to_air_rl.automation.strategies.balanced import BalancedStrategy


# Mock classes for testing
class MockPosition:
    def __init__(self, lat=45.0, lon=-120.0, alt=10000.0):
        self.lat = lat
        self.lon = lon
        self.alt = alt


class MockVelocity:
    def __init__(self, vx=100.0, vy=100.0, vz=0.0):
        self.vx = vx
        self.vy = vy
        self.vz = vz


class MockRadar:
    def __init__(self):
        self.max_range_m = 60000
        self.h_fov_deg = 60


class MockAircraft:
    def __init__(self, aircraft_id="test_aircraft"):
        self.id = aircraft_id
        self.group = "blue"
        self.position = MockPosition()
        self.velocity = MockVelocity()
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.speed = 300.0
        self.max_speed_mps = 500.0
        self.min_speed_mps = 150.0
        self.min_alt_m = 500.0
        self.max_alt_m = 20000.0
        self.stall0_mps = 60.0

        # Systems
        self.radar = MockRadar()

        # Weapons
        self.remaining_missiles = 4
        self.missiles = []

        # Countermeasures
        self.flares = 20
        self.chaff = 20
        self.ecm = 1
        self.decoys = 2

        # Map limits
        self.map_limits = MockMapLimits()


class MockMapLimits:
    def __init__(self):
        self.top_lat = 46.0
        self.bottom_lat = 44.0
        self.left_lon = -121.0
        self.right_lon = -119.0


class MockTarget:
    def __init__(self, target_id="enemy1"):
        self.id = target_id
        self.group = "red"
        self.position = MockPosition(45.1, -119.9, 9000)  # Different position
        self.velocity = MockVelocity(50, 50, 0)
        self.yaw_deg = 180.0  # Heading toward us
        self.speed = 250.0
        self.remaining_missiles = 2


class MockSimulator:
    def __init__(self):
        self.utc_time = 0.0
        self.active_units = {}


def test_behavior_tree_nodes():
    """Test basic behavior tree node functionality."""
    print("Testing behavior tree nodes...")

    # Test selector node
    selector = SelectorNode("TestSelector")

    # Add condition nodes
    true_condition = ConditionNode("AlwaysTrue", lambda ctx: True)
    false_condition = ConditionNode("AlwaysFalse", lambda ctx: False)

    selector.add_child(false_condition)
    selector.add_child(true_condition)

    # Test execution
    result = selector.tick({})
    assert result == NodeStatus.SUCCESS, f"Expected SUCCESS, got {result}"

    print("[PASS] Behavior tree nodes test")


def test_energy_manager():
    """Test energy management system."""
    print("Testing energy management...")

    aircraft = MockAircraft()
    energy_manager = EnergyManager(aircraft)

    # Test specific energy calculation
    energy = energy_manager.calculate_specific_energy()
    assert energy > 0, "Specific energy should be positive"

    # Test energy advantage calculation
    target = MockTarget()
    energy_manager.assess_energy_advantage(target)

    # Test flight profile calculation
    context = {"tactical_situation": "defensive"}
    altitude, speed, throttle = energy_manager.calculate_optimal_flight_profile(context)

    assert 0 <= throttle <= 1.0, f"Throttle should be 0-1, got {throttle}"
    assert altitude > 0, f"Altitude should be positive, got {altitude}"
    assert speed > 0, f"Speed should be positive, got {speed}"

    print("[PASS] Energy management test")


def test_geometry_calculator():
    """Test geometry calculations."""
    print("Testing geometry calculator...")

    aircraft = MockAircraft()
    target = MockTarget()
    geometry_calc = GeometryCalculator(aircraft)

    # Test aspect angle calculation
    aspect_angle = geometry_calc.calculate_aspect_angle(target)
    assert 0 <= aspect_angle <= 180, f"Aspect angle should be 0-180, got {aspect_angle}"

    # Test angle off tail
    aot = geometry_calc.calculate_angle_off_tail(target)
    assert 0 <= aot <= 180, f"Angle off tail should be 0-180, got {aot}"

    # Test antenna train angle
    ata = geometry_calc.calculate_antenna_train_angle(target)
    assert ata >= 0, f"ATA should be positive, got {ata}"

    # Test engagement envelope
    envelope = geometry_calc.calculate_engagement_envelope(target)
    assert "aspect_angle_deg" in envelope
    assert "tactical_advantage" in envelope

    print("[PASS] Geometry calculator test")


def test_range_calculator():
    """Test range calculations."""
    print("Testing range calculator...")

    aircraft = MockAircraft()
    target = MockTarget()
    range_calc = RangeCalculator(aircraft)

    # Test NEZ calculation
    nez_ranges = range_calc.calculate_nez_ranges(target)
    assert "nez_in_m" in nez_ranges
    assert "nez_out_m" in nez_ranges
    assert nez_ranges["nez_out_m"] > nez_ranges["nez_in_m"]

    # Test F-pole calculation
    f_pole = range_calc.calculate_f_pole_range(target)
    assert f_pole >= 0, f"F-pole should be non-negative, got {f_pole}"

    # Test launch envelope
    envelope = range_calc.calculate_launch_envelope(target)
    assert envelope["valid"]
    assert "engagement_zone" in envelope
    assert "shot_quality" in envelope

    print("[PASS] Range calculator test")


def test_bvr_tactics():
    """Test BVR tactical maneuvers."""
    print("Testing BVR tactics...")

    aircraft = MockAircraft()
    target = MockTarget()
    bvr_tactics = BVRTactics(aircraft)

    # Test crank maneuver
    context = {"target": target}
    result = bvr_tactics.execute_crank_maneuver(context)
    assert result == NodeStatus.SUCCESS
    assert "desired_yaw" in context
    assert "desired_throttle" in context

    # Test approach maneuver
    context = {"target": target}
    result = bvr_tactics.execute_approach(context)
    assert result == NodeStatus.SUCCESS

    print("[PASS] BVR tactics test")


def test_full_tactical_controller():
    """Test the complete tactical controller."""
    print("Testing full tactical controller...")

    aircraft = MockAircraft()
    config = BalancedStrategy.create_config()
    controller = FullTacticalController(aircraft, config)

    # Test action generation
    simulator = MockSimulator()
    simulator.active_units[aircraft.id] = aircraft

    # Add a target
    target = MockTarget()
    simulator.active_units[target.id] = target

    action = controller.get_action(simulator, 0.1)

    # Validate action format (10-dim for energy+lift-vector space)
    assert len(action) == 10, f"Action should have 10 elements, got {len(action)}"
    assert all(0 <= a <= 1 for a in action), "All action values should be 0-1"

    # Test status reporting
    status = controller.get_status()
    assert "tactical_state" in status
    assert "behavior_tree_active" in status
    assert "flight_controls" in status

    print("[PASS] Full tactical controller test")


def test_tactical_context_building():
    """Test tactical context building."""
    print("Testing tactical context building...")

    aircraft = MockAircraft()
    controller = FullTacticalController(aircraft, BalancedStrategy.create_config())

    simulator = MockSimulator()
    target = MockTarget()
    simulator.active_units[target.id] = target

    # Set a target for context building
    controller.target = target

    context = controller._build_tactical_context(simulator, 0.1)

    # Validate context structure
    required_keys = ["aircraft", "target", "threats", "geometry", "ranges", "energy"]
    for key in required_keys:
        assert key in context, f"Context missing required key: {key}"

    # Validate derived information
    if context["target"]:
        assert "target_range_m" in context
        assert "in_launch_range" in context

    print("[PASS] Tactical context building test")


def test_action_space_conversion():
    """Test conversion between desired values and action space."""
    print("Testing action space conversion...")

    aircraft = MockAircraft()
    controller = FullTacticalController(aircraft, BalancedStrategy.create_config())

    # Test yaw conversion
    desired_yaw = 90.0  # Turn right
    action_yaw = controller._yaw_to_action_space(desired_yaw)
    assert 0 <= action_yaw <= 1, f"Yaw action should be 0-1, got {action_yaw}"

    # Test pitch conversion
    desired_pitch = 10.0  # Climb
    action_pitch = controller._pitch_to_action_space(desired_pitch)
    assert 0 <= action_pitch <= 1, f"Pitch action should be 0-1, got {action_pitch}"

    print("[PASS] Action space conversion test")


def test_integration():
    """Test complete system integration."""
    print("Testing system integration...")

    aircraft = MockAircraft()
    controller = FullTacticalController(aircraft, BalancedStrategy.create_config())

    simulator = MockSimulator()
    simulator.active_units[aircraft.id] = aircraft

    # Add target
    target = MockTarget()
    simulator.active_units[target.id] = target

    # Run multiple decision cycles
    for i in range(5):
        action = controller.get_action(simulator, 0.1)

        # Validate each action (10-dim for energy+lift-vector space)
        assert len(action) == 10
        assert all(isinstance(a, (int, float, np.integer, np.floating)) for a in action)

        # Update time
        simulator.utc_time += 0.1

    # Check that controller maintained state
    status = controller.get_status()
    assert status["behavior_tree_active"]

    print("[PASS] System integration test")


def run_all_tests():
    """Run all scripted control tests."""
    print("Running scripted tactical controller tests...\n")

    try:
        test_behavior_tree_nodes()
        test_energy_manager()
        test_geometry_calculator()
        test_range_calculator()
        test_bvr_tactics()
        test_full_tactical_controller()
        test_tactical_context_building()
        test_action_space_conversion()
        test_integration()

        print("\n[SUCCESS] All scripted control tests passed!")
        print("\nThe fully scripted tactical controller is ready for combat!")

        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
