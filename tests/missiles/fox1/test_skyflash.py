from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestSkyflash:
    """Test the Skyflash missile."""

    @pytest.fixture
    def mock_source(self):
        """Create a mock source unit (aircraft)."""
        source = Mock()
        source.position = Mock()
        source.position.lat = 45.0
        source.position.lon = 2.0
        source.position.alt = 10000.0
        source.position.copy.return_value = source.position
        source.yaw_deg = 45.0
        source.speed = 300.0
        source.pitch_deg = 5.0
        source.group = "BLUE"
        source.name = "Eurofighter_Blue"
        return source

    @pytest.fixture
    def mock_target(self):
        """Create a mock target unit."""
        target = Mock()
        target.position = Mock()
        target.position.lat = 46.0
        target.position.lon = 3.0
        target.position.alt = 9000.0
        target.name = "Su27_Red"
        target.group = "RED"
        return target

    @pytest.fixture
    def mock_map_limits(self):
        """Create mock map limits."""
        return {"lat_min": 44.0, "lat_max": 47.0, "lon_min": 1.0, "lon_max": 4.0}

    def test_skyflash_import(self):
        """Test that Skyflash missile can be imported."""
        from air_to_air_rl.missiles.fox1.skyflash import Skyflash

        assert Skyflash is not None

    def test_skyflash_initialization(self, mock_source, mock_target, mock_map_limits):
        """Test Skyflash missile initialization."""
        from air_to_air_rl.missiles.fox1.skyflash import Skyflash

        with patch(
            "air_to_air_rl.missiles.fox1.base_fox1.Fox1Missile.__init__", return_value=None
        ) as mock_init:
            Skyflash(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
            )

            # Verify initialization was called with correct parameters
            mock_init.assert_called_once()
            args, kwargs = mock_init.call_args

            assert kwargs["name"] == "Skyflash"
            assert kwargs["firing_time_s"] == 0.0
            assert kwargs["target"] == mock_target
            assert kwargs["source"] == mock_source
            assert kwargs["map_limits"] == mock_map_limits
            assert kwargs["group"] == "BLUE"

            # Check configuration values
            config = kwargs["config"]
            assert config["mass_kg"] == 193.0
            assert config["max_speed_mps"] == 1300.0
            assert config["motor_burn_s"] == 9.0
            assert config["life_time_s"] == 65.0
            assert config["hit_probability"] == 0.88
            assert config["n_max"] == 28.0

    def test_skyflash_radar_configuration(self, mock_source, mock_target, mock_map_limits):
        """Test Skyflash radar configuration."""
        from air_to_air_rl.missiles.fox1.skyflash import Skyflash

        with patch(
            "air_to_air_rl.missiles.fox1.base_fox1.Fox1Missile.__init__", return_value=None
        ) as mock_init:
            Skyflash(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
            )

            # Check radar configuration
            config = mock_init.call_args[1]["config"]
            radar_config = config["radar"]

            assert radar_config["fov_deg"] == 22.0
            assert radar_config["max_range_m"] == 50000.0
            assert radar_config["sensitivity"] == 1.3

    def test_skyflash_vs_sparrow_differences(self, mock_source, mock_target, mock_map_limits):
        """Test differences between Skyflash and Sparrow configurations."""
        from air_to_air_rl.missiles.fox1.skyflash import Skyflash
        from air_to_air_rl.missiles.fox1.sparrow import AIM7_Sparrow

        with patch(
            "air_to_air_rl.missiles.fox1.base_fox1.Fox1Missile.__init__", return_value=None
        ) as mock_init:
            # Test Skyflash
            Skyflash(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
            )
            skyflash_config = mock_init.call_args[1]["config"]

            mock_init.reset_mock()

            # Test Sparrow
            AIM7_Sparrow(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
            )
            sparrow_config = mock_init.call_args[1]["config"]

            # Compare key differences
            assert skyflash_config["mass_kg"] < sparrow_config["mass_kg"]  # Skyflash is lighter
            assert (
                skyflash_config["motor_burn_s"] > sparrow_config["motor_burn_s"]
            )  # Longer burn time
            assert skyflash_config["life_time_s"] > sparrow_config["life_time_s"]  # Longer range
            assert (
                skyflash_config["radar"]["fov_deg"] < sparrow_config["radar"]["fov_deg"]
            )  # Narrower FOV

    def test_skyflash_performance_characteristics(self, mock_source, mock_target, mock_map_limits):
        """Test Skyflash performance characteristics."""
        from air_to_air_rl.missiles.fox1.skyflash import Skyflash

        with patch(
            "air_to_air_rl.missiles.fox1.base_fox1.Fox1Missile.__init__", return_value=None
        ) as mock_init:
            Skyflash(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
            )

            config = mock_init.call_args[1]["config"]

            # Test aerodynamic characteristics
            assert config["reference_area_m2"] == 0.18
            assert config["aspect_ratio"] == 3.2
            assert config["oswald_e"] == 0.78
            assert config["drag_coefficient"] == 0.11

            # Test propulsion characteristics
            assert config["constant_engine_N"] == 11000.0

            # Test guidance characteristics
            assert config["min_range_m"] == 800.0
            assert config["seeker_sensitivity"] == 1.3
