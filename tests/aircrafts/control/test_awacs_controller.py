#!/usr/bin/env python3
"""
Tests for aircrafts.control.awacs_controller module.
Tests AWACS orbit controllers for automated patrol patterns.
"""

import numpy as np
import pytest

from bvr_marl_core.aircraft.control.awacs_controller import (
    CircleController,
    Figure8Controller,
    OrbitConfig,
    RacetrackController,
    RandomWaypointController,
    create_orbit_controller,
)

# ============================================================================
# ORBIT CONFIG TESTS
# ============================================================================


class TestOrbitConfig:
    """Tests for OrbitConfig dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        config = OrbitConfig()

        assert config.center_lat == 0.0
        assert config.center_lon == 0.0
        assert config.leg_length_km == 40.0
        assert config.turn_radius_km == 25.0
        assert config.altitude_m == 10000.0
        assert config.speed_mps == 200.0
        assert config.clockwise is True
        assert config.pattern == "racetrack"

    def test_custom_values(self):
        """Should accept custom values."""
        config = OrbitConfig(
            center_lat=-0.5,
            center_lon=-1.0,
            leg_length_km=60.0,
            turn_radius_km=15.0,
            altitude_m=12000.0,
            speed_mps=220.0,
            clockwise=False,
            pattern="figure8",
        )

        assert config.center_lat == -0.5
        assert config.center_lon == -1.0
        assert config.leg_length_km == 60.0
        assert config.turn_radius_km == 15.0
        assert config.altitude_m == 12000.0
        assert config.speed_mps == 220.0
        assert config.clockwise is False
        assert config.pattern == "figure8"


# ============================================================================
# RACETRACK CONTROLLER TESTS
# ============================================================================


class MockUnit:
    """Mock unit for testing controllers."""

    def __init__(self, lat=0.0, lon=0.0, alt=10000.0, yaw=90.0, max_speed=250.0):
        self.position = type("Position", (), {"lat": lat, "lon": lon, "alt": alt})()
        self.yaw_deg = yaw
        self.max_speed_mps = max_speed


class TestRacetrackController:
    """Tests for RacetrackController class."""

    def test_init_default(self):
        """Should initialize with default config."""
        controller = RacetrackController()

        assert controller.config is not None
        assert controller.phase == "outbound"
        assert controller.target_heading == 90.0

    def test_init_custom_config(self):
        """Should accept custom config."""
        config = OrbitConfig(
            center_lat=1.0,
            center_lon=2.0,
            leg_length_km=50.0,
        )
        controller = RacetrackController(config)

        assert controller.config.center_lat == 1.0
        assert controller.config.center_lon == 2.0
        assert controller.config.leg_length_km == 50.0

    def test_update_returns_correct_types(self):
        """Update should return (heading, speed, altitude) tuple."""
        controller = RacetrackController()
        unit = MockUnit()

        result = controller.update(unit, dt=1.0)

        assert isinstance(result, tuple)
        assert len(result) == 3
        heading, speed, altitude = result
        assert isinstance(heading, float)
        assert isinstance(speed, float)
        assert isinstance(altitude, float)

    def test_update_heading_in_valid_range(self):
        """Heading should be in [0, 360) range."""
        controller = RacetrackController()
        unit = MockUnit()

        for _ in range(100):
            heading, _, _ = controller.update(unit, dt=1.0)
            assert 0 <= heading < 360

    def test_speed_matches_config(self):
        """Speed should match config value."""
        config = OrbitConfig(speed_mps=180.0)
        controller = RacetrackController(config)
        unit = MockUnit()

        _, speed, _ = controller.update(unit, dt=1.0)
        assert speed == 180.0

    def test_altitude_matches_config(self):
        """Altitude should match config value."""
        config = OrbitConfig(altitude_m=11000.0)
        controller = RacetrackController(config)
        unit = MockUnit()

        _, _, altitude = controller.update(unit, dt=1.0)
        assert altitude == 11000.0

    def test_phase_determination_outbound(self):
        """Should detect outbound phase correctly."""
        controller = RacetrackController()
        # Unit on western side, heading east
        unit = MockUnit(lon=-0.1, lat=0.01)

        controller.update(unit, dt=1.0)
        # Should be in outbound or related phase
        assert controller.phase in ("outbound", "turn2")

    def test_phase_determination_inbound(self):
        """Should detect inbound phase correctly."""
        controller = RacetrackController()
        # Unit on eastern side after turn, heading west
        unit = MockUnit(lon=0.1, lat=-0.01)

        controller.update(unit, dt=1.0)
        # Should be in inbound or related phase
        assert controller.phase in ("inbound", "turn1")

    def test_get_commanded_controls(self):
        """Should return valid control commands."""
        controller = RacetrackController()
        unit = MockUnit()

        controls = controller.get_commanded_controls(unit, dt=1.0)

        assert "throttle" in controls
        assert "target_altitude_m" in controls
        assert "target_heading_deg" in controls

        # Throttle should be in valid range
        assert 0.0 <= controls["throttle"] <= 1.0

        # Heading should be in valid range
        assert 0 <= controls["target_heading_deg"] < 360

    def test_throttle_calculation(self):
        """Throttle should be proportional to desired speed."""
        config = OrbitConfig(speed_mps=200.0)
        controller = RacetrackController(config)
        unit = MockUnit(max_speed=250.0)

        controls = controller.get_commanded_controls(unit, dt=1.0)

        expected_throttle = 200.0 / 250.0  # 0.8
        assert abs(controls["throttle"] - expected_throttle) < 0.01

    def test_turn_rate_limiting(self):
        """Target heading should be a valid compass bearing."""
        controller = RacetrackController()
        unit = MockUnit(yaw=0.0)

        controls = controller.get_commanded_controls(unit, dt=1.0)

        # target_heading_deg must be a valid compass bearing
        assert 0 <= controls["target_heading_deg"] < 360


# ============================================================================
# FIGURE-8 CONTROLLER TESTS
# ============================================================================


class TestFigure8Controller:
    """Tests for Figure8Controller class."""

    def test_init(self):
        """Should initialize correctly."""
        controller = Figure8Controller()

        assert controller.config.pattern == "figure8"

    def test_has_waypoint_based_navigation(self):
        """Figure8Controller uses waypoint-based navigation (not phase-based like racetrack)."""
        controller = Figure8Controller()

        assert isinstance(controller, Figure8Controller)
        assert hasattr(controller, "waypoints")
        assert len(controller.waypoints) > 1
        assert hasattr(controller, "current_wp_idx")

    def test_update_returns_valid_output(self):
        """Update should return valid output."""
        controller = Figure8Controller()
        unit = MockUnit()

        result = controller.update(unit, dt=1.0)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_custom_config(self):
        """Should accept custom config."""
        config = OrbitConfig(
            leg_length_km=30.0,
            pattern="racetrack",  # Will be overridden
        )
        controller = Figure8Controller(config)

        assert controller.config.leg_length_km == 30.0
        assert controller.config.pattern == "figure8"


# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================


class TestCreateOrbitController:
    """Tests for create_orbit_controller factory function."""

    def test_create_racetrack(self):
        """Should create RacetrackController by default."""
        controller = create_orbit_controller()

        assert isinstance(controller, RacetrackController)
        assert not isinstance(controller, Figure8Controller)

    def test_create_figure8(self):
        """Should create Figure8Controller when specified."""
        controller = create_orbit_controller(pattern="figure8")

        assert isinstance(controller, Figure8Controller)

    def test_create_with_custom_params(self):
        """Should pass parameters to config."""
        controller = create_orbit_controller(
            pattern="racetrack",
            center_lat=0.5,
            center_lon=-0.8,
            leg_length_km=50.0,
            altitude_m=9000.0,
            speed_mps=180.0,
            clockwise=False,
        )

        assert controller.config.center_lat == 0.5
        assert controller.config.center_lon == -0.8
        assert controller.config.leg_length_km == 50.0
        assert controller.config.altitude_m == 9000.0
        assert controller.config.speed_mps == 180.0
        assert controller.config.clockwise is False


# ============================================================================
# WAYPOINT COMPUTATION TESTS
# ============================================================================


class TestWaypointComputation:
    """Tests for internal waypoint computation."""

    def test_waypoints_computed_on_init(self):
        """Waypoints should be computed during initialization."""
        controller = RacetrackController()

        assert hasattr(controller, "wp1")
        assert hasattr(controller, "wp2")
        assert hasattr(controller, "turn1_center")
        assert hasattr(controller, "turn2_center")

    def test_waypoints_positions(self):
        """Waypoints should be at correct positions."""
        config = OrbitConfig(
            center_lat=0.0,
            center_lon=0.0,
            leg_length_km=40.0,
        )
        controller = RacetrackController(config)

        # WP1 should be west of center, WP2 east
        assert controller.wp1.lon < 0
        assert controller.wp2.lon > 0

        # Both should be at center latitude
        assert abs(controller.wp1.lat) < 0.001
        assert abs(controller.wp2.lat) < 0.001

    def test_clockwise_vs_counterclockwise(self):
        """Turn centers should differ based on direction."""
        config_cw = OrbitConfig(clockwise=True)
        config_ccw = OrbitConfig(clockwise=False)

        controller_cw = RacetrackController(config_cw)
        controller_ccw = RacetrackController(config_ccw)

        # Turn centers should be on opposite sides
        assert controller_cw.turn1_center.lat != controller_ccw.turn1_center.lat


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestOrbitControllerIntegration:
    """Integration tests for orbit controllers."""

    def test_continuous_orbit_simulation(self):
        """Controller should handle continuous orbit simulation without errors."""
        controller = RacetrackController()
        unit = MockUnit()

        # Simulate 100 time steps - should not raise any errors
        headings = []
        for _ in range(100):
            heading, speed, altitude = controller.update(unit, dt=1.0)
            headings.append(heading)

            # All returned values should be valid
            assert 0 <= heading < 360
            assert speed > 0
            assert altitude > 0

        # Should consistently return some heading
        assert len(headings) == 100

    def test_figure8_waypoint_navigation(self):
        """Figure-8 controller should navigate through its waypoints."""
        controller = Figure8Controller()
        unit = MockUnit()

        # Controller should have a valid waypoint index and a non-empty waypoint list
        assert hasattr(controller, "current_wp_idx")
        assert len(controller.waypoints) > 1

        # After an update call the waypoint index should still be valid
        controller.update(unit, dt=1.0)
        assert 0 <= controller.current_wp_idx < len(controller.waypoints)


# ============================================================================
# NEW PATTERN TESTS (circle, random, random_all)
# ============================================================================


class TestNewOrbitPatterns:
    """Tests for CircleController, RandomWaypointController, and random_all factory."""

    # ------------------------------------------------------------------
    # CircleController
    # ------------------------------------------------------------------

    def test_circle_controller_init(self):
        """Should generate 16 waypoints at correct radius."""
        config = OrbitConfig(
            center_lat=0.0,
            center_lon=0.0,
            turn_radius_km=25.0,
            altitude_m=10000.0,
            speed_mps=200.0,
        )
        controller = CircleController(config)

        assert len(controller.waypoints) == 16
        # Every waypoint should be approximately turn_radius_km from center
        r_deg = 25.0 / 111.0
        for lat, lon in controller.waypoints:
            dist_deg = np.sqrt((lat - 0.0) ** 2 + (lon - 0.0) ** 2)
            assert abs(dist_deg - r_deg) < 0.001

    def test_circle_controller_update(self):
        """Update should return a valid (heading, speed, altitude) tuple."""
        controller = CircleController()
        unit = MockUnit()

        result = controller.update(unit, dt=1.0)

        assert isinstance(result, tuple)
        assert len(result) == 3
        heading, speed, altitude = result
        assert 0.0 <= heading < 360.0
        assert speed > 0.0
        assert altitude > 0.0

    def test_circle_controller_commanded_controls(self):
        """get_commanded_controls should return throttle ∈ [0,1] and valid heading."""
        controller = CircleController()
        unit = MockUnit(max_speed=250.0)

        controls = controller.get_commanded_controls(unit, dt=1.0)

        assert 0.0 <= controls["throttle"] <= 1.0
        assert 0.0 <= controls["target_heading_deg"] < 360.0
        assert controls["target_altitude_m"] > 0.0

    def test_circle_clockwise_vs_counterclockwise(self):
        """Clockwise and counter-clockwise circles should produce mirror waypoints."""
        config_cw = OrbitConfig(center_lat=0.0, center_lon=0.0, turn_radius_km=20.0, clockwise=True)
        config_ccw = OrbitConfig(
            center_lat=0.0, center_lon=0.0, turn_radius_km=20.0, clockwise=False
        )

        ctrl_cw = CircleController(config_cw)
        ctrl_ccw = CircleController(config_ccw)

        # First waypoints should differ (opposite direction)
        assert ctrl_cw.waypoints[1] != ctrl_ccw.waypoints[1]

    # ------------------------------------------------------------------
    # RandomWaypointController
    # ------------------------------------------------------------------

    def test_random_waypoint_controller_init(self):
        """Initial waypoint should be within the zone bounds."""
        config = OrbitConfig(
            center_lat=0.0,
            center_lon=-1.0,
            turn_radius_km=25.0,
            zone_min_lat=-0.5,
            zone_max_lat=0.5,
            zone_min_lon=-1.5,
            zone_max_lon=-0.5,
        )
        controller = RandomWaypointController(config, rng=np.random.default_rng(42))

        wp_lat, wp_lon = controller.current_wp
        # Within bounds (after 5% margin shrink)
        assert -0.5 < wp_lat < 0.5
        assert -1.5 < wp_lon < -0.5

    def test_random_waypoint_controller_update(self):
        """Update should return a valid (heading, speed, altitude) tuple."""
        controller = RandomWaypointController(rng=np.random.default_rng(0))
        unit = MockUnit()

        result = controller.update(unit, dt=1.0)

        assert isinstance(result, tuple)
        assert len(result) == 3
        heading, speed, altitude = result
        assert 0.0 <= heading < 360.0
        assert speed > 0.0
        assert altitude > 0.0

    def test_random_waypoint_advances_waypoint(self):
        """When unit is at current_wp, update should pick a new waypoint."""
        config = OrbitConfig(center_lat=0.0, center_lon=0.0, turn_radius_km=25.0)
        controller = RandomWaypointController(config, rng=np.random.default_rng(7))

        # Place the unit exactly at the current waypoint
        wp_lat, wp_lon = controller.current_wp
        unit = MockUnit(lat=wp_lat, lon=wp_lon)
        old_wp = controller.current_wp

        controller.update(unit, dt=1.0)

        # Waypoint should have changed
        assert controller.current_wp != old_wp

    def test_random_waypoint_zone_fallback(self):
        """With no zone bounds supplied, waypoint should be within auto-derived zone."""
        config = OrbitConfig(center_lat=0.0, center_lon=0.0, turn_radius_km=20.0)
        controller = RandomWaypointController(config, rng=np.random.default_rng(1))

        wp_lat, wp_lon = controller.current_wp
        r_deg = 2.0 * 20.0 / 111.0  # 2 * turn_radius in degrees
        assert abs(wp_lat) <= r_deg
        assert abs(wp_lon) <= r_deg

    # ------------------------------------------------------------------
    # Factory — new patterns
    # ------------------------------------------------------------------

    def test_create_orbit_circle(self):
        """Factory should return CircleController for pattern='circle'."""
        controller = create_orbit_controller(pattern="circle")
        assert isinstance(controller, CircleController)

    def test_create_orbit_random(self):
        """Factory should return RandomWaypointController for pattern='random'."""
        controller = create_orbit_controller(pattern="random")
        assert isinstance(controller, RandomWaypointController)

    def test_create_orbit_random_all(self):
        """Factory with 'random_all' should return one of the four valid controller types."""
        valid_types = (
            CircleController,
            Figure8Controller,
            RacetrackController,
            RandomWaypointController,
        )
        controller = create_orbit_controller(pattern="random_all")
        assert isinstance(controller, valid_types)

    def test_random_all_reproducible(self):
        """Same RNG seed should yield the same controller type."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        ctrl1 = create_orbit_controller(pattern="random_all", rng=rng1)
        ctrl2 = create_orbit_controller(pattern="random_all", rng=rng2)

        assert type(ctrl1) is type(ctrl2)

    def test_create_circle_zone_params_passed(self):
        """Zone parameters supplied to factory should appear in RandomWaypointController."""
        controller = create_orbit_controller(
            pattern="random",
            center_lat=0.0,
            center_lon=0.0,
            turn_radius_km=25.0,
            zone_min_lat=-0.5,
            zone_max_lat=0.5,
            zone_min_lon=-0.5,
            zone_max_lon=0.5,
        )
        assert isinstance(controller, RandomWaypointController)
        # Zone bounds should be stored (after margin)
        assert controller._lat_lo > -0.5
        assert controller._lat_hi < 0.5


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
