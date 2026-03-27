"""
Tests for scenario configuration (AWACS and regime randomization).
"""

import pytest

from air_to_air_rl.rl.environment.gym.gym_components.config import (
    AWACSConfigData,
    BVREnvConfig,
    GeometryConfigData,
    ScenarioConfigData,
)


class TestAWACSConfigData:
    """Tests for AWACSConfigData dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        config = AWACSConfigData()

        # AWACS disabled by default
        assert config.agent_awacs is False
        assert config.opponent_awacs is False

        # Positioning defaults
        assert config.orbit_distance_km == 80.0
        assert config.orbit_altitude_m == 10000.0
        assert config.orbit_speed_mps == 200.0

        # Orbit pattern defaults
        assert config.orbit_pattern == "figure8"
        assert config.orbit_leg_length_km == 40.0
        assert config.orbit_clockwise is True

        # Lock FOV default
        assert config.lock_fov_deg == 60.0

    def test_custom_values(self):
        """Should accept custom values."""
        config = AWACSConfigData(
            agent_awacs=True,
            opponent_awacs=True,
            orbit_distance_km=100.0,
            orbit_altitude_m=12000.0,
            orbit_speed_mps=180.0,
            orbit_pattern="figure8",
            orbit_leg_length_km=50.0,
            orbit_clockwise=False,
            lock_fov_deg=90.0,
        )

        assert config.agent_awacs is True
        assert config.opponent_awacs is True
        assert config.orbit_distance_km == 100.0
        assert config.orbit_altitude_m == 12000.0
        assert config.orbit_speed_mps == 180.0
        assert config.orbit_pattern == "figure8"
        assert config.orbit_leg_length_km == 50.0
        assert config.orbit_clockwise is False
        assert config.lock_fov_deg == 90.0


class TestScenarioConfigData:
    """Tests for ScenarioConfigData dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        config = ScenarioConfigData()

        assert config.allowed_regimes == []
        assert config.randomize_regime is True
        assert isinstance(config.awacs_config, AWACSConfigData)

    def test_with_allowed_regimes(self):
        """Should accept list of allowed regimes."""
        config = ScenarioConfigData(
            allowed_regimes=["hot_80km", "hot_120km", "offset_30deg"],
            randomize_regime=True,
        )

        assert len(config.allowed_regimes) == 3
        assert "hot_80km" in config.allowed_regimes
        assert "hot_120km" in config.allowed_regimes
        assert "offset_30deg" in config.allowed_regimes

    def test_with_awacs_config(self):
        """Should accept AWACSConfigData."""
        awacs = AWACSConfigData(agent_awacs=True, opponent_awacs=True)
        config = ScenarioConfigData(awacs_config=awacs)

        assert config.awacs_config.agent_awacs is True
        assert config.awacs_config.opponent_awacs is True


class TestBVREnvConfigScenario:
    """Tests for scenario config integration in BVREnvConfig."""

    def test_from_dict_default_scenario_config(self):
        """Should have default scenario config when not specified."""
        config_dict = {
            "num_agents_per_side": 2,
            "map_size": 200,
        }
        config = BVREnvConfig.from_dict(config_dict)

        assert hasattr(config, "scenario_config")
        assert isinstance(config.scenario_config, ScenarioConfigData)
        assert config.scenario_config.allowed_regimes == []
        assert config.scenario_config.awacs_config.agent_awacs is False

    def test_from_dict_with_scenario_config(self):
        """Should parse scenario_config from dict."""
        config_dict = {
            "num_agents_per_side": 2,
            "map_size": 200,
            "scenario_config": {
                "allowed_regimes": ["hot_80km", "hot_120km"],
                "randomize_regime": True,
                "awacs_config": {
                    "agent_awacs": True,
                    "opponent_awacs": False,
                    "orbit_distance_km": 100.0,
                    "lock_fov_deg": 90.0,
                },
            },
        }
        config = BVREnvConfig.from_dict(config_dict)

        assert config.scenario_config.allowed_regimes == ["hot_80km", "hot_120km"]
        assert config.scenario_config.randomize_regime is True
        assert config.scenario_config.awacs_config.agent_awacs is True
        assert config.scenario_config.awacs_config.opponent_awacs is False
        assert config.scenario_config.awacs_config.orbit_distance_km == 100.0
        assert config.scenario_config.awacs_config.lock_fov_deg == 90.0

    def test_from_dict_awacs_all_options(self):
        """Should parse all AWACS options."""
        config_dict = {
            "num_agents_per_side": 2,
            "map_size": 200,
            "scenario_config": {
                "awacs_config": {
                    "agent_awacs": True,
                    "opponent_awacs": True,
                    "orbit_distance_km": 120.0,
                    "orbit_altitude_m": 11000.0,
                    "orbit_speed_mps": 190.0,
                    "orbit_pattern": "figure8",
                    "orbit_leg_length_km": 50.0,
                    "orbit_clockwise": False,
                    "lock_fov_deg": 150.0,
                },
            },
        }
        config = BVREnvConfig.from_dict(config_dict)

        awacs = config.scenario_config.awacs_config
        assert awacs.agent_awacs is True
        assert awacs.opponent_awacs is True
        assert awacs.orbit_distance_km == 120.0
        assert awacs.orbit_altitude_m == 11000.0
        assert awacs.orbit_speed_mps == 190.0
        assert awacs.orbit_pattern == "figure8"
        assert awacs.orbit_leg_length_km == 50.0
        assert awacs.orbit_clockwise is False
        assert awacs.lock_fov_deg == 150.0


class TestScenarioWithGeometry:
    """Tests for scenario config interaction with geometry config."""

    def test_allowed_regimes_overrides_use_controlled_geometry(self):
        """allowed_regimes should take precedence over use_controlled_geometry."""
        config_dict = {
            "num_agents_per_side": 2,
            "map_size": 200,
            "geometry_config": {
                "use_controlled_geometry": False,
                "regime": "hot_80km",
            },
            "scenario_config": {
                "allowed_regimes": ["hot_120km", "offset_30deg"],
            },
        }
        config = BVREnvConfig.from_dict(config_dict)

        # geometry_config should still be parsed
        assert config.geometry_config.use_controlled_geometry is False
        assert config.geometry_config.regime == "hot_80km"

        # But scenario_config has the allowed_regimes
        assert config.scenario_config.allowed_regimes == ["hot_120km", "offset_30deg"]

    def test_empty_allowed_regimes_falls_back_to_geometry(self):
        """Empty allowed_regimes should allow geometry_config to control spawning."""
        config_dict = {
            "num_agents_per_side": 2,
            "map_size": 200,
            "geometry_config": {
                "use_controlled_geometry": True,
                "regime": "hot_160km",
            },
            "scenario_config": {
                "allowed_regimes": [],
            },
        }
        config = BVREnvConfig.from_dict(config_dict)

        assert config.geometry_config.use_controlled_geometry is True
        assert config.geometry_config.regime == "hot_160km"
        assert config.scenario_config.allowed_regimes == []


class TestScenarioConfigValidation:
    """Tests for scenario config validation and edge cases."""

    def test_single_regime_no_randomization_needed(self):
        """Single regime doesn't need randomization."""
        config = ScenarioConfigData(
            allowed_regimes=["hot_120km"],
            randomize_regime=True,  # Still set but won't matter
        )
        assert len(config.allowed_regimes) == 1

    def test_awacs_one_side_only(self):
        """Should allow AWACS on only one side."""
        config = AWACSConfigData(
            agent_awacs=True,
            opponent_awacs=False,
        )
        assert config.agent_awacs is True
        assert config.opponent_awacs is False

    def test_awacs_both_sides(self):
        """Should allow AWACS on both sides."""
        config = AWACSConfigData(
            agent_awacs=True,
            opponent_awacs=True,
        )
        assert config.agent_awacs is True
        assert config.opponent_awacs is True

    def test_awacs_neither_side(self):
        """Should allow no AWACS on either side."""
        config = AWACSConfigData(
            agent_awacs=False,
            opponent_awacs=False,
        )
        assert config.agent_awacs is False
        assert config.opponent_awacs is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
