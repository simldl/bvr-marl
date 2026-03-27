from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestAIM9Sidewinder:
    """Test the AIM-9 Sidewinder missile."""

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
        source.name = "F16_Blue"
        return source

    @pytest.fixture
    def mock_target(self):
        """Create a mock target unit."""
        target = Mock()
        target.position = Mock()
        target.position.lat = 46.0
        target.position.lon = 3.0
        target.position.alt = 9000.0
        target.name = "MiG21_Red"
        target.group = "RED"
        target.temperature = 450.0  # Hot target for IR tracking
        return target

    @pytest.fixture
    def mock_map_limits(self):
        """Create mock map limits."""
        return {"lat_min": 44.0, "lat_max": 47.0, "lon_min": 1.0, "lon_max": 4.0}

    def test_sidewinder_import(self):
        """Test that Sidewinder missile can be imported."""
        from air_to_air_rl.missiles.fox2.sidewinder import AIM9_Sidewinder

        assert AIM9_Sidewinder is not None

    def test_sidewinder_initialization(self, mock_source, mock_target, mock_map_limits):
        """Test Sidewinder missile initialization."""
        from air_to_air_rl.missiles.fox2.sidewinder import AIM9_Sidewinder

        with patch(
            "air_to_air_rl.missiles.fox2.base_fox2.Fox2Missile.__init__", return_value=None
        ) as mock_init:
            AIM9_Sidewinder(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
            )

            # Verify initialization was called with correct parameters
            mock_init.assert_called_once()
            args, kwargs = mock_init.call_args

            assert kwargs["name"] == "AIM9_Sidewinder"
            assert kwargs["firing_time_s"] == 0.0
            assert kwargs["target"] == mock_target
            assert kwargs["source"] == mock_source
            assert kwargs["map_limits"] == mock_map_limits
            assert kwargs["group"] == "BLUE"

            # Check configuration values - lightweight, agile missile
            config = kwargs["config"]
            assert config["mass_kg"] == 85.0
            assert config["max_speed_mps"] == 900.0
            assert config["motor_burn_s"] == 6.0  # Short burn time
            assert config["life_time_s"] == 45.0  # Short range
            assert config["hit_probability"] == 0.90  # High accuracy at close range
            assert config["n_max"] == 35.0  # High maneuverability

    def test_sidewinder_ir_seeker_configuration(self, mock_source, mock_target, mock_map_limits):
        """Test Sidewinder IR seeker configuration."""
        from air_to_air_rl.missiles.fox2.sidewinder import AIM9_Sidewinder

        with patch(
            "air_to_air_rl.missiles.fox2.base_fox2.Fox2Missile.__init__", return_value=None
        ) as mock_init:
            AIM9_Sidewinder(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
            )

            # Check IR seeker configuration
            config = mock_init.call_args[1]["config"]
            radar_config = config["radar"]  # Actually IR seeker config

            assert radar_config["fov_deg"] == 15.0  # Narrow FOV for precision
            assert radar_config["max_range_m"] == 15000.0  # Short range
            assert radar_config["sensitivity"] == 1.0

    def test_sidewinder_performance_characteristics(
        self, mock_source, mock_target, mock_map_limits
    ):
        """Test Sidewinder performance characteristics."""
        from air_to_air_rl.missiles.fox2.sidewinder import AIM9_Sidewinder

        with patch(
            "air_to_air_rl.missiles.fox2.base_fox2.Fox2Missile.__init__", return_value=None
        ) as mock_init:
            AIM9_Sidewinder(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
            )

            config = mock_init.call_args[1]["config"]

            # Test aerodynamic characteristics - optimized for agility
            assert config["reference_area_m2"] == 0.15
            assert config["aspect_ratio"] == 2.8
            assert config["oswald_e"] == 0.8
            assert config["drag_coefficient"] == 0.08  # Low drag

            # Test propulsion characteristics - short but powerful
            assert config["constant_engine_N"] == 8000.0
            assert config["motor_burn_s"] == 6.0

            # Test guidance characteristics - close range, high precision
            assert config["min_range_m"] == 500.0  # Very short minimum range
            assert config["seeker_sensitivity"] == 1.0

    def test_sidewinder_vs_other_missiles_comparison(
        self, mock_source, mock_target, mock_map_limits
    ):
        """Test how Sidewinder compares to other missile types."""
        from air_to_air_rl.missiles.fox2.sidewinder import AIM9_Sidewinder

        with patch(
            "air_to_air_rl.missiles.fox2.base_fox2.Fox2Missile.__init__", return_value=None
        ) as mock_init:
            AIM9_Sidewinder(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
            )

            config = mock_init.call_args[1]["config"]

            # Sidewinder characteristics vs typical FOX1/FOX3
            # - Much lighter weight
            assert config["mass_kg"] < 100.0  # Much lighter than FOX1/3

            # - Higher maneuverability
            assert config["n_max"] >= 35.0  # Higher G capability

            # - Shorter range but higher accuracy
            assert config["life_time_s"] < 60.0  # Shorter flight time
            assert config["hit_probability"] >= 0.90  # Higher hit probability

            # - Smaller minimum engagement range
            assert config["min_range_m"] <= 1000.0  # Better for close combat

    def test_sidewinder_heat_signature_requirements(
        self, mock_source, mock_target, mock_map_limits
    ):
        """Test Sidewinder heat signature tracking requirements."""
        from air_to_air_rl.missiles.fox2.sidewinder import AIM9_Sidewinder

        with patch(
            "air_to_air_rl.missiles.fox2.base_fox2.Fox2Missile.__init__", return_value=None
        ) as mock_init:
            AIM9_Sidewinder(
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
            )

            config = mock_init.call_args[1]["config"]

            # Sidewinder should be optimized for heat tracking
            # The radar config is actually IR seeker config for FOX2
            ir_config = config["radar"]

            # Should have reasonable FOV for heat source acquisition
            assert 10.0 <= ir_config["fov_deg"] <= 20.0

            # Should have sensitivity appropriate for heat tracking
            assert ir_config["sensitivity"] > 0.0
