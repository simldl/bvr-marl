from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestPython:
    """Test the Python missile."""

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
        target.name = "Mirage_Red"
        target.group = "RED"
        target.temperature = 480.0  # Hot target for IR tracking
        return target

    @pytest.fixture
    def mock_map_limits(self):
        """Create mock map limits."""
        return {"lat_min": 44.0, "lat_max": 47.0, "lon_min": 1.0, "lon_max": 4.0}

    def test_python_import(self):
        """Test that Python missile can be imported."""
        from air_to_air_rl.missiles.fox2.python import Python_5

        assert Python_5 is not None

    def test_python_initialization(self, mock_source, mock_target, mock_map_limits):
        """Test Python missile initialization."""
        from air_to_air_rl.missiles.fox2.python import Python_5

        with patch(
            "air_to_air_rl.missiles.fox2.base_fox2.Fox2Missile.__init__", return_value=None
        ) as mock_init:
            Python_5(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
            )

            # Verify initialization was called with correct parameters
            mock_init.assert_called_once()
            args, kwargs = mock_init.call_args

            assert kwargs["name"] == "Python_5"
            assert kwargs["firing_time_s"] == 0.0
            assert kwargs["target"] == mock_target
            assert kwargs["source"] == mock_source
            assert kwargs["map_limits"] == mock_map_limits
            assert kwargs["group"] == "BLUE"

            # Check configuration values - European equivalent to Sidewinder
            config = kwargs["config"]
            assert config["mass_kg"] == 100.0  # Slightly heavier than Sidewinder
            assert config["max_speed_mps"] == 950.0  # Slightly faster
            assert config["motor_burn_s"] == 7.0  # Slightly longer burn
            assert config["life_time_s"] == 40.0  # Range
            assert config["hit_probability"] == 0.92  # High accuracy
            assert config["n_max"] == 40.0  # High maneuverability

    def test_python_vs_sidewinder_differences(self, mock_source, mock_target, mock_map_limits):
        """Test differences between Python and Sidewinder configurations."""
        from air_to_air_rl.missiles.fox2.python import Python_5
        from air_to_air_rl.missiles.fox2.sidewinder import AIM9_Sidewinder

        with patch(
            "air_to_air_rl.missiles.fox2.base_fox2.Fox2Missile.__init__", return_value=None
        ) as mock_init:
            # Test Python
            Python_5(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
            )
            python_config = mock_init.call_args[1]["config"]

            mock_init.reset_mock()

            # Test Sidewinder
            AIM9_Sidewinder(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
            )
            sidewinder_config = mock_init.call_args[1]["config"]

            # Compare key differences
            assert (
                python_config["mass_kg"] > sidewinder_config["mass_kg"]
            )  # Python slightly heavier
            assert (
                python_config["max_speed_mps"] > sidewinder_config["max_speed_mps"]
            )  # Python faster
            assert python_config["motor_burn_s"] > sidewinder_config["motor_burn_s"]  # Longer burn
            assert (
                python_config["life_time_s"] < sidewinder_config["life_time_s"]
            )  # Shorter range but faster

            # Python has higher hit probability and maneuverability
            assert (
                python_config["hit_probability"] > sidewinder_config["hit_probability"]
            )  # Python has better hit probability
            assert (
                python_config["n_max"] > sidewinder_config["n_max"]
            )  # Python has higher G capability

    def test_python_ir_seeker_configuration(self, mock_source, mock_target, mock_map_limits):
        """Test Python IR seeker configuration."""
        from air_to_air_rl.missiles.fox2.python import Python_5

        with patch(
            "air_to_air_rl.missiles.fox2.base_fox2.Fox2Missile.__init__", return_value=None
        ) as mock_init:
            Python_5(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
            )

            # Check IR seeker configuration
            config = mock_init.call_args[1]["config"]
            radar_config = config["radar"]  # Actually IR seeker config

            assert radar_config["fov_deg"] == 18.0  # Slightly wider than Sidewinder
            assert radar_config["max_range_m"] == 20000.0  # Longer range than Sidewinder
            assert radar_config["sensitivity"] == 1.2  # More sensitive

    def test_python_performance_characteristics(self, mock_source, mock_target, mock_map_limits):
        """Test Python performance characteristics."""
        from air_to_air_rl.missiles.fox2.python import Python_5

        with patch(
            "air_to_air_rl.missiles.fox2.base_fox2.Fox2Missile.__init__", return_value=None
        ) as mock_init:
            Python_5(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
            )

            config = mock_init.call_args[1]["config"]

            # Test aerodynamic characteristics
            assert config["reference_area_m2"] == 0.16  # Slightly larger than Sidewinder
            assert config["aspect_ratio"] == 2.6
            assert config["oswald_e"] == 0.82
            assert config["drag_coefficient"] == 0.07

            # Test propulsion characteristics
            assert config["constant_engine_N"] == 9000.0  # More powerful than Sidewinder
            assert config["motor_burn_s"] == 7.0

            # Test guidance characteristics
            assert config["min_range_m"] == 400.0  # Shorter minimum range
            assert config["seeker_sensitivity"] == 1.2

    def test_python_european_design_philosophy(self, mock_source, mock_target, mock_map_limits):
        """Test Python missile European design philosophy."""
        from air_to_air_rl.missiles.fox2.python import Python_5

        with patch(
            "air_to_air_rl.missiles.fox2.base_fox2.Fox2Missile.__init__", return_value=None
        ) as mock_init:
            Python_5(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
            )

            config = mock_init.call_args[1]["config"]

            # European design typically favors longer range over extreme maneuverability
            assert config["life_time_s"] == 40.0  # Good range
            assert config["max_speed_mps"] >= 950.0  # Good speed
            assert config["hit_probability"] >= 0.85  # Good accuracy

            # Balanced approach between range and agility
            assert config["n_max"] == 40.0  # High maneuverability
