import numpy as np
import pytest

from air_to_air_rl.rl.environment.gym.gym_components.config import AWACSConfigData
from air_to_air_rl.rl.environment.gym.spawn_utils import (
    MapBoundaryChecker,
    compute_awacs_orbit_center,
    spawn_position,
    spawn_unit,
)
from air_to_air_rl.simulator.core.helpers import Position


class DummyMapLimits:
    left_lon = -1
    right_lon = 1
    bottom_lat = -1
    top_lat = 1
    min_alt = 1000
    max_alt = 12000


class DummyAircraft:
    def __init__(self, pos, yaw, speed, group, map_limits, min_alt, max_alt):
        self.position = pos
        self.yaw_deg = yaw
        self.speed = speed
        self.group = group


class DummySim:
    def __init__(self):
        self.units = []
        self.calls = []

    def add_unit(self, unit):
        self.units.append(unit)
        return len(self.units)

    def record_unit_trace(self, uid):
        self.calls.append(uid)


class DummyEnv:
    def __init__(self):
        self.aircraft_type_map = {"A": DummyAircraft, "B": DummyAircraft}
        self.map_size_km = 10
        self.map_limits = DummyMapLimits()
        self.simulator = DummySim()
        self.config = {"default_speed": 300}


def test_spawn_position_agent_and_opponent():
    limits = DummyMapLimits()
    pos_agent = spawn_position("agent", 10, limits)
    pos_op = spawn_position("opponent", 10, limits)
    assert limits.left_lon <= pos_agent.lon <= limits.right_lon
    assert limits.min_alt <= pos_agent.alt <= limits.max_alt
    assert limits.left_lon <= pos_op.lon <= limits.right_lon


def test_spawn_unit_registers():
    env = DummyEnv()
    uid = spawn_unit(env, "A", "agent")
    assert uid == 1
    assert isinstance(env.simulator.units[0], DummyAircraft)


def test_map_boundary_checker():
    u = type("U", (), {"position": type("P", (), {"lon": 0, "lat": 0})()})()
    limits = DummyMapLimits()
    assert MapBoundaryChecker.within_bounds(u, limits)
    u.position.lon = 2
    assert not MapBoundaryChecker.within_bounds(u, limits)


# ============================================================================
# AWACS Orbit Center Computation Tests
# ============================================================================


class TestComputeAwacsOrbitCenter:
    """Tests for compute_awacs_orbit_center function."""

    def test_agent_awacs_positioned_west_of_fighters(self):
        """Agent AWACS should be positioned west (more negative lon) than fighters."""
        center_lon = compute_awacs_orbit_center(
            group="agent",
            geometry_data=None,
            map_size_km=200.0,
            orbit_distance_km=80.0,
        )
        # Agent fighters are on west side, AWACS should be even further west
        assert center_lon < 0

    def test_opponent_awacs_positioned_east_of_fighters(self):
        """Opponent AWACS should be positioned east (more positive lon) than fighters."""
        center_lon = compute_awacs_orbit_center(
            group="opponent",
            geometry_data=None,
            map_size_km=200.0,
            orbit_distance_km=80.0,
        )
        # Opponent fighters are on east side, AWACS should be even further east
        assert center_lon > 0

    def test_orbit_distance_affects_position(self):
        """Larger orbit distance should place AWACS further from center."""
        center_near = compute_awacs_orbit_center(
            group="agent",
            geometry_data=None,
            map_size_km=200.0,
            orbit_distance_km=40.0,
        )
        center_far = compute_awacs_orbit_center(
            group="agent",
            geometry_data=None,
            map_size_km=200.0,
            orbit_distance_km=120.0,
        )
        # Further orbit distance = more negative longitude for agent
        assert center_far < center_near

    def test_with_geometry_data_agent(self):
        """Should use geometry data positions when available."""
        geometry_data = {
            "agent_positions": [
                Position(lat=0.0, lon=-0.3, alt=8000),
                Position(lat=0.0, lon=-0.35, alt=8000),
            ],
            "opponent_positions": [
                Position(lat=0.0, lon=0.3, alt=8000),
            ],
        }
        center_lon = compute_awacs_orbit_center(
            group="agent",
            geometry_data=geometry_data,
            map_size_km=200.0,
            orbit_distance_km=80.0,
        )
        # Should be west of average agent position (-0.325)
        assert center_lon < -0.325

    def test_with_geometry_data_opponent(self):
        """Should use geometry data positions when available for opponents."""
        geometry_data = {
            "agent_positions": [
                Position(lat=0.0, lon=-0.3, alt=8000),
            ],
            "opponent_positions": [
                Position(lat=0.0, lon=0.3, alt=8000),
                Position(lat=0.0, lon=0.35, alt=8000),
            ],
        }
        center_lon = compute_awacs_orbit_center(
            group="opponent",
            geometry_data=geometry_data,
            map_size_km=200.0,
            orbit_distance_km=80.0,
        )
        # Should be east of average opponent position (0.325)
        assert center_lon > 0.325


class TestAWACSConfigData:
    """Tests for AWACSConfigData usage in spawning."""

    def test_default_config(self):
        """Default config should have AWACS disabled."""
        config = AWACSConfigData()
        assert config.agent_awacs is False
        assert config.opponent_awacs is False

    def test_orbit_pattern_options(self):
        """Should support different orbit patterns."""
        config_racetrack = AWACSConfigData(orbit_pattern="racetrack")
        config_figure8 = AWACSConfigData(orbit_pattern="figure8")

        assert config_racetrack.orbit_pattern == "racetrack"
        assert config_figure8.orbit_pattern == "figure8"

    def test_lock_fov_separate_from_detection(self):
        """Lock FOV should be configurable separately."""
        config = AWACSConfigData(lock_fov_deg=90.0)
        assert config.lock_fov_deg == 90.0

        # Detection FOV is set in AWACS.Config, not here
        # This just configures what lock_fov_deg to pass to AWACS


class TestAWACSNonEngageable:
    """Tests for the awacs_non_engageable flag in AWACSConfigData."""

    def test_awacs_non_engageable_default_is_true(self):
        """awacs_non_engageable must default to True (support-asset semantics)."""
        config = AWACSConfigData()
        assert config.awacs_non_engageable is True

    def test_awacs_non_engageable_can_be_disabled(self):
        """awacs_non_engageable can be set to False (AWACS becomes a valid target)."""
        config = AWACSConfigData(awacs_non_engageable=False)
        assert config.awacs_non_engageable is False

    def test_awacs_non_engageable_independent_of_awacs_enabled(self):
        """Flag is present regardless of whether AWACS are enabled in the scenario."""
        config_on = AWACSConfigData(opponent_awacs=True, awacs_non_engageable=True)
        config_off = AWACSConfigData(opponent_awacs=False, awacs_non_engageable=True)
        assert config_on.awacs_non_engageable is True
        assert config_off.awacs_non_engageable is True
