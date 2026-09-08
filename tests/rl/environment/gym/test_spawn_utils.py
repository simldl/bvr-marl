import numpy as np
import pytest

from bvr_marl_core.rl.environment.gym.gym_components.config import AWACSConfigData, BVREnvConfig
from bvr_marl_core.rl.environment.gym.spawn_utils import (
    MapBoundaryChecker,
    compute_awacs_orbit_center,
    spawn_awacs,
    spawn_position,
    spawn_unit,
    spawn_unit_with_geometry,
)
from bvr_marl_core.simulator.core.helpers import Position


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
        self.active_units = {}
        self.calls = []

    def add_unit(self, unit):
        self.units.append(unit)
        uid = len(self.units)
        unit.id = uid
        self.active_units[uid] = unit
        return uid

    def record_unit_trace(self, uid):
        self.calls.append(uid)


class DummyEnv:
    def __init__(self):
        self.aircraft_type_map = {"A": DummyAircraft, "B": DummyAircraft}
        self.map_size_km = 10
        self.map_limits = DummyMapLimits()
        self.simulator = DummySim()
        self.config = {"default_speed": 300}


def _dummy_awacs_env(config_dict: dict):
    bvr_config = BVREnvConfig.from_dict(config_dict)
    env = type("AwacsEnv", (), {})()
    env.config = {}
    env.map_size_km = bvr_config.map_size_km
    env.map_limits = bvr_config.map_limits
    env.full_map_limits = bvr_config.full_map_limits
    env.simulator = DummySim()
    return env, bvr_config.scenario_config.awacs_config


def test_spawn_position_agent_and_opponent():
    limits = DummyMapLimits()
    pos_agent = spawn_position("agent", 10, limits)
    pos_op = spawn_position("opponent", 10, limits)
    assert limits.left_lon <= pos_agent.lon <= limits.right_lon
    assert limits.min_alt <= pos_agent.alt <= limits.max_alt
    assert limits.left_lon <= pos_op.lon <= limits.right_lon


def test_spawn_position_respects_rectangular_limits():
    class RectLimits(DummyMapLimits):
        left_lon = -2
        right_lon = 2
        bottom_lat = -0.5
        top_lat = 0.5

    limits = RectLimits()
    pos_agent = spawn_position("agent", 400, limits)
    pos_op = spawn_position("opponent", 400, limits)

    assert limits.bottom_lat <= pos_agent.lat <= limits.top_lat
    assert limits.bottom_lat <= pos_op.lat <= limits.top_lat
    assert limits.left_lon <= pos_agent.lon <= 0.0
    assert 0.0 <= pos_op.lon <= limits.right_lon


def test_spawn_unit_registers():
    env = DummyEnv()
    uid = spawn_unit(env, "A", "agent")
    assert uid == 1
    assert isinstance(env.simulator.units[0], DummyAircraft)


def test_spawn_unit_marks_anchored_opponent_boundary_kept():
    env = DummyEnv()
    env.config = {
        "default_speed": 300,
        "scenario_config": {"opponent_behavior": "anchored_hold"},
    }

    uid = spawn_unit(env, "B", "opponent")
    unit = env.simulator.active_units[uid]

    assert unit.keep_inside_boundary is True
    assert unit.scripted_anchor_position is not unit.position
    assert unit.scripted_anchor_position.lat == pytest.approx(unit.position.lat)
    assert unit.scripted_anchor_position.lon == pytest.approx(unit.position.lon)


def test_spawn_unit_marks_stationary_opponent_boundary_kept():
    env = DummyEnv()
    env.config = {
        "default_speed": 300,
        "scenario_config": {"opponent_behavior": "stationary_hold"},
    }

    uid = spawn_unit(env, "B", "opponent")
    unit = env.simulator.active_units[uid]

    assert unit.keep_inside_boundary is True


def test_spawn_unit_with_geometry_marks_anchored_opponent_boundary_kept():
    env = DummyEnv()
    env.config = {"scenario_config": {"opponent_behavior": "anchored_hold"}}
    pos = Position(lat=0.9, lon=0.9, alt=8000.0)

    uid = spawn_unit_with_geometry(env, "B", "opponent", pos, heading=270.0, speed=250.0)
    unit = env.simulator.active_units[uid]

    assert unit.keep_inside_boundary is True
    assert unit.scripted_anchor_position.lat == pytest.approx(0.9)
    assert unit.scripted_anchor_position.lon == pytest.approx(0.9)


def test_spawn_unit_with_geometry_marks_stationary_opponent_boundary_kept():
    env = DummyEnv()
    env.config = {"scenario_config": {"opponent_behavior": "stationary_hold"}}
    pos = Position(lat=0.9, lon=0.9, alt=8000.0)

    uid = spawn_unit_with_geometry(env, "B", "opponent", pos, heading=270.0, speed=250.0)
    unit = env.simulator.active_units[uid]

    assert unit.keep_inside_boundary is True
    assert unit.scripted_anchor_position.lat == pytest.approx(0.9)
    assert unit.scripted_anchor_position.lon == pytest.approx(0.9)


def test_policy_opponent_is_not_boundary_clamped_by_spawn():
    env = DummyEnv()
    env.config = {"scenario_config": {"opponent_behavior": "policy"}}

    uid = spawn_unit(env, "B", "opponent")
    unit = env.simulator.active_units[uid]

    assert not getattr(unit, "keep_inside_boundary", False)


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
        assert center_lon == pytest.approx(-(80.0 / 111.0))

    def test_opponent_awacs_positioned_east_of_fighters(self):
        """Opponent AWACS should be positioned east (more positive lon) than fighters."""
        center_lon = compute_awacs_orbit_center(
            group="opponent",
            geometry_data=None,
            map_size_km=200.0,
            orbit_distance_km=80.0,
        )
        assert center_lon == pytest.approx(80.0 / 111.0)

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
        """Geometry data does not affect the fixed side-placement helper."""
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
        assert center_lon == pytest.approx(-(80.0 / 111.0))

    def test_with_geometry_data_opponent(self):
        """Geometry data does not affect the fixed side-placement helper."""
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
        assert center_lon == pytest.approx(80.0 / 111.0)


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

    def test_fixed_side_zone_orbit_is_default(self):
        """AWACS should not chase fighter geometry unless explicitly configured."""
        config = AWACSConfigData()
        assert config.trail_fighters is False

    def test_spawn_awacs_uses_configured_fixed_orbit_center(self):
        env, awacs_config = _dummy_awacs_env(
            {
                "num_agents_per_side": 1,
                "map_size": 160,
                "scenario_config": {
                    "awacs_config": {
                        "agent_awacs": True,
                        "orbit_distance_km": 80.0,
                        "orbit_radius_km": 25.0,
                    },
                },
            }
        )

        uid = spawn_awacs(env, "agent", awacs_config)
        awacs = env.simulator.active_units[uid]
        orbit_config = awacs.orbit_controller.config

        assert awacs.position.lat == pytest.approx(0.0)
        assert awacs.position.lon == pytest.approx(-(80.0 / 111.0))
        assert orbit_config.center_lat == pytest.approx(0.0)
        assert orbit_config.center_lon == pytest.approx(-(80.0 / 111.0))
        assert orbit_config.trail_fighters is False
        assert awacs.map_limits.right_lon == pytest.approx(105.0 / 111.0)


class TestAWACSSplit:
    """Tests for the two-less-potent-AWACS split (count_per_team, radar overrides)."""

    def test_default_is_two_less_potent_awacs(self):
        config = AWACSConfigData()
        assert config.count_per_team == 2

    def test_placements_are_symmetric_north_south(self):
        from bvr_marl_core.rl.environment.gym.gym_components.episode_manager import (
            _awacs_placements,
        )

        placements = _awacs_placements(AWACSConfigData(awacs_pair_spacing_km=120.0))
        assert len(placements) == 2
        (lat_lo, suf_lo), (lat_hi, suf_hi) = placements
        assert lat_lo == pytest.approx(-120.0 / 111.0 / 2.0)
        assert lat_hi == pytest.approx(+120.0 / 111.0 / 2.0)
        assert suf_lo == "_1" and suf_hi == "_2"

    def test_single_awacs_sits_on_axis(self):
        from bvr_marl_core.rl.environment.gym.gym_components.episode_manager import (
            _awacs_placements,
        )

        assert _awacs_placements(AWACSConfigData(count_per_team=1)) == [(0.0, "")]

    def test_awacs_radar_defaults_to_less_potent_250km(self):
        env, awacs_config = _dummy_awacs_env(
            {"scenario_config": {"awacs_config": {"agent_awacs": True}}}
        )
        uid = spawn_awacs(env, "agent", awacs_config)
        awacs = env.simulator.active_units[uid]
        assert awacs.radar.max_range_m == pytest.approx(250_000.0)
        assert awacs.radar.h_fov_deg == pytest.approx(360.0)

    def test_radar_overrides_apply(self):
        env, awacs_config = _dummy_awacs_env(
            {"scenario_config": {"awacs_config": {"agent_awacs": True}}}
        )
        uid = spawn_awacs(
            env,
            "agent",
            awacs_config,
            radar_overrides={"radar_max_range_m": 300_000.0},
        )
        awacs = env.simulator.active_units[uid]
        assert awacs.radar.max_range_m == pytest.approx(300_000.0)


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
