"""
Smoke tests for the scripted/automated tactical controller.

Verifies that:
- The scripted controller components can be imported
- The FullTacticalController can be instantiated with mock aircraft
- Core tactical submodules (behavior tree, BVR tactics, geometry, energy)
  produce valid outputs when called with lightweight mock objects
"""

import pytest

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Minimal mock objects (no simulator needed)
# ---------------------------------------------------------------------------


class _MockPosition:
    def __init__(self, lat=45.0, lon=-120.0, alt=10000.0):
        self.lat = lat
        self.lon = lon
        self.alt = alt


class _MockRadar:
    max_range_m = 60_000
    h_fov_deg = 60


class _MockAircraft:
    id = "test_ac"
    group = "blue"
    yaw_deg = 90.0
    speed = 250.0
    pitch_deg = 0.0
    min_speed_mps = 80.0
    max_speed_mps = 600.0
    stall0_mps = 60.0
    n_max = 9.0
    flares = 10
    chaff = 10
    ecm = 2
    decoys = 2
    missiles = []
    remaining_missiles = 4

    def __init__(self):
        self.position = _MockPosition()
        self.radar = _MockRadar()

    class _MapLimits:
        top_lat = 46.0
        bottom_lat = 44.0
        left_lon = -121.0
        right_lon = -119.0

    map_limits = _MapLimits()


class _MockTarget:
    id = "enemy1"
    group = "red"
    yaw_deg = 180.0
    speed = 250.0

    def __init__(self):
        self.position = _MockPosition(45.1, -119.9, 9000.0)


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


def test_scripted_controller_imports():
    """All scripted controller modules can be imported without error."""
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
    from air_to_air_rl.automation.scripted_control.tactics.geometry_calculator import (
        GeometryCalculator,
    )
    from air_to_air_rl.automation.scripted_control.tactics.range_calculator import RangeCalculator


def test_behavior_tree_nodes():
    """Behavior tree selector/condition nodes execute without error."""
    from air_to_air_rl.automation.scripted_control.behavior_tree.nodes import (
        ConditionNode,
        NodeStatus,
        SelectorNode,
    )

    selector = SelectorNode("TestSelector")
    selector.add_child(ConditionNode("AlwaysFalse", lambda _ctx: False))
    selector.add_child(ConditionNode("AlwaysTrue", lambda _ctx: True))

    result = selector.tick({})
    assert result == NodeStatus.SUCCESS


def test_energy_manager_specific_energy():
    """EnergyManager.calculate_specific_energy() returns a positive value."""
    from air_to_air_rl.automation.scripted_control.tactics.energy_management import EnergyManager

    manager = EnergyManager(_MockAircraft())
    energy = manager.calculate_specific_energy()
    assert energy > 0


def test_energy_manager_flight_profile():
    """EnergyManager.calculate_optimal_flight_profile() returns valid control targets."""
    from air_to_air_rl.automation.scripted_control.tactics.energy_management import EnergyManager

    manager = EnergyManager(_MockAircraft())
    altitude, speed, throttle = manager.calculate_optimal_flight_profile(
        {"tactical_situation": "offensive"}
    )
    assert 0.0 <= throttle <= 1.0
    assert altitude > 0
    assert speed > 0


def test_geometry_calculator_aspect_angle():
    """GeometryCalculator computes a valid aspect angle for a mock target."""
    from air_to_air_rl.automation.scripted_control.tactics.geometry_calculator import (
        GeometryCalculator,
    )

    calc = GeometryCalculator(_MockAircraft())
    angle = calc.calculate_aspect_angle(_MockTarget())
    assert 0.0 <= angle <= 180.0


def test_geometry_calculator_engagement_envelope():
    """GeometryCalculator.calculate_engagement_envelope returns expected keys."""
    from air_to_air_rl.automation.scripted_control.tactics.geometry_calculator import (
        GeometryCalculator,
    )

    calc = GeometryCalculator(_MockAircraft())
    envelope = calc.calculate_engagement_envelope(_MockTarget())
    assert "aspect_angle_deg" in envelope
    assert "tactical_advantage" in envelope


def test_bvr_tactics_crank_maneuver():
    """BVRTactics.execute_crank_maneuver completes and injects desired_yaw into context."""
    from air_to_air_rl.automation.scripted_control.behavior_tree.nodes import NodeStatus
    from air_to_air_rl.automation.scripted_control.tactics.bvr_tactics import BVRTactics

    tactics = BVRTactics(_MockAircraft())
    context: dict = {"target": _MockTarget()}
    result = tactics.execute_crank_maneuver(context)
    assert result == NodeStatus.SUCCESS
    assert "desired_yaw" in context
    assert "desired_throttle" in context


def test_full_tactical_controller_instantiation():
    """FullTacticalController can be instantiated without a live simulator."""
    from air_to_air_rl.automation.scripted_control.full_tactical_controller import (
        FullTacticalController,
    )

    # No simulator means action_processor is None — that's fine for smoke test
    controller = FullTacticalController(_MockAircraft(), simulator=None)
    assert controller is not None
    assert controller.bvr_tactics is not None
    assert controller.energy_manager is not None
    assert controller.geometry_calc is not None
    assert controller.range_calc is not None
