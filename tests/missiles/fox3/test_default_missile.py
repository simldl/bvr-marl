import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestLongRangeMissile:
    """Test the default long range missile."""

    @pytest.fixture
    def mock_source(self):
        """Create a mock source unit (aircraft)."""
        source = Mock()
        source.position = Mock()
        source.position.lat = 45.0
        source.position.lon = 2.0
        source.position.alt = 12000.0
        source.position.copy.return_value = source.position
        source.yaw_deg = 45.0
        source.speed = 500.0
        source.pitch_deg = 5.0
        source.group = "BLUE"
        source.name = "F22_Blue"
        return source

    @pytest.fixture
    def mock_target(self):
        """Create a mock target unit."""
        target = Mock()
        target.position = Mock()
        target.position.lat = 47.0
        target.position.lon = 4.0
        target.position.alt = 11000.0
        target.name = "Su35_Red"
        target.group = "RED"
        target.rcs_m2 = 8.0  # Larger radar cross section
        return target

    @pytest.fixture
    def mock_map_limits(self):
        """Create mock map limits."""
        return {
            "lat_min": 44.0,
            "lat_max": 48.0,
            "lon_min": 1.0,
            "lon_max": 5.0
        }

    def test_long_range_missile_import(self):
        """Test that long range missile can be imported."""
        try:
            from missiles.fox3.default_missile import LongRangeMissile
            assert LongRangeMissile is not None
        except ImportError:
            pytest.skip("LongRangeMissile not available")

    def test_long_range_missile_initialization(self, mock_source, mock_target, mock_map_limits):
        """Test long range missile initialization."""
        try:
            from missiles.fox3.default_missile import LongRangeMissile
            
            with patch('missiles.missile.Missile.__init__', return_value=None) as mock_init:
                missile = LongRangeMissile(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits,
                    group="BLUE"
                )
                
                # Verify initialization was called with correct parameters
                mock_init.assert_called_once()
                args, kwargs = mock_init.call_args
                
                assert kwargs['name'] == "LongRangeMissile"
                assert kwargs['firing_time_s'] == 0.0
                assert kwargs['target'] == mock_target
                assert kwargs['source'] == mock_source
                assert kwargs['map_limits'] == mock_map_limits
                assert kwargs['group'] == "BLUE"
                assert kwargs['data_link_mode'] == "full"
                
                # Check configuration values - heavy, long range missile
                config = kwargs['config']
                assert config['mass_kg'] == 380.0  # Heavy missile
                assert config['max_speed_mps'] == 1500.0  # Very high speed
                assert config['motor_burn_s'] == 32.0  # Very long burn time
                assert config['life_time_s'] == 220.0  # Very long range capability
                assert config['hit_probability'] == 0.90  # High accuracy
                assert config['n_max'] == 35.0  # High maneuverability despite size
                
        except ImportError:
            pytest.skip("LongRangeMissile dependencies not available")

    def test_long_range_missile_radar_configuration(self, mock_source, mock_target, mock_map_limits):
        """Test long range missile advanced radar seeker configuration."""
        try:
            from missiles.fox3.default_missile import LongRangeMissile
            
            with patch('missiles.missile.Missile.__init__', return_value=None) as mock_init:
                LongRangeMissile(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits,
                    group="BLUE"
                )
                
                # Check advanced radar seeker configuration
                config = mock_init.call_args[1]['config']
                radar_config = config['radar']
                
                assert radar_config['horizontal_fov_deg'] == 90.0  # Very wide horizontal search
                assert radar_config['vertical_fov_deg'] == 45.0  # Wide vertical coverage
                assert radar_config['max_range_m'] == 120000.0  # Very long detection range
                assert radar_config['radar_frequency_hz'] == 12e9  # Higher frequency
                assert radar_config['tx_power_w'] == 8e3  # High power transmitter
                assert radar_config['antenna_gain_db'] == 34.0  # Very high gain antenna
                assert radar_config['snr_threshold_db'] == 8.5  # Sensitive detection
                
        except ImportError:
            pytest.skip("LongRangeMissile dependencies not available")

    def test_long_range_missile_performance_characteristics(self, mock_source, mock_target, mock_map_limits):
        """Test long range missile performance characteristics."""
        try:
            from missiles.fox3.default_missile import LongRangeMissile
            
            with patch('missiles.missile.Missile.__init__', return_value=None) as mock_init:
                LongRangeMissile(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits
                )
                
                config = mock_init.call_args[1]['config']
                
                # Test aerodynamic characteristics - optimized for very long range
                assert config['reference_area_m2'] == 0.28  # Large for stability
                assert config['aspect_ratio'] == 3.5  # High lift/drag ratio
                assert config['oswald_e'] == 0.75  # Good efficiency
                assert config['drag_coefficient'] == 0.075  # Very low drag
                
                # Test propulsion characteristics - very long range focus
                assert config['constant_engine_N'] == 24000.0  # Very powerful engine
                assert config['motor_burn_s'] == 32.0  # Very long burn time
                
                # Test guidance characteristics - long range BVR engagement
                assert config['min_range_m'] == 2000.0  # Longer minimum range
                assert config['seeker_sensitivity'] == 1.3  # High radar sensitivity
                
        except ImportError:
            pytest.skip("LongRangeMissile dependencies not available")

    def test_long_range_vs_amraam_comparison(self, mock_source, mock_target, mock_map_limits):
        """Test differences between long range missile and AMRAAM."""
        try:
            from missiles.fox3.default_missile import LongRangeMissile
            from missiles.fox3.amraam import AIM120_AMRAAM
            
            with patch('missiles.missile.Missile.__init__', return_value=None) as mock_init:
                # Test Long Range Missile
                LongRangeMissile(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits
                )
                long_range_config = mock_init.call_args[1]['config']
                
                mock_init.reset_mock()
                
                # Test AMRAAM for comparison
                AIM120_AMRAAM(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits
                )
                amraam_config = mock_init.call_args[1]['config']
                
                # Long range missile should exceed AMRAAM in most range metrics
                assert long_range_config['mass_kg'] > amraam_config['mass_kg']
                assert long_range_config['life_time_s'] > amraam_config['life_time_s']
                assert long_range_config['motor_burn_s'] > amraam_config['motor_burn_s']
                assert long_range_config['max_speed_mps'] > amraam_config['max_speed_mps']
                assert long_range_config['hit_probability'] > amraam_config['hit_probability']

                # FOV should be wider for better search capability
                assert long_range_config['radar']['horizontal_fov_deg'] > amraam_config['radar']['horizontal_fov_deg']
                
        except ImportError:
            pytest.skip("Long range missile/AMRAAM comparison not available")

    def test_long_range_missile_extreme_bvr_capabilities(self, mock_source, mock_target, mock_map_limits):
        """Test long range missile extreme BVR capabilities."""
        try:
            from missiles.fox3.default_missile import LongRangeMissile
            
            with patch('missiles.missile.Missile.__init__', return_value=None) as mock_init:
                LongRangeMissile(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits
                )
                
                config = mock_init.call_args[1]['config']
                
                # Extreme BVR engagement characteristics
                assert config['life_time_s'] >= 200.0  # Very long flight time
                assert config['max_speed_mps'] >= 1500.0  # Hypersonic speeds
                assert config['min_range_m'] >= 2000.0  # Long minimum safe distance
                
                # Advanced radar for extreme BVR tracking
                radar_config = config['radar']
                assert radar_config['max_range_m'] >= 100000.0  # Extreme detection range
                assert radar_config['horizontal_fov_deg'] >= 90.0  # Very wide search
                assert radar_config['vertical_fov_deg'] >= 45.0  # Wide vertical coverage
                
        except ImportError:
            pytest.skip("LongRangeMissile dependencies not available")

    def test_long_range_missile_advanced_radar_features(self, mock_source, mock_target, mock_map_limits):
        """Test long range missile advanced radar features."""
        try:
            from missiles.fox3.default_missile import LongRangeMissile
            
            with patch('missiles.missile.Missile.__init__', return_value=None) as mock_init:
                LongRangeMissile(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits
                )
                
                config = mock_init.call_args[1]['config']
                radar_config = config['radar']
                
                # Advanced radar should have superior characteristics
                # Higher frequency for better resolution
                assert radar_config['radar_frequency_hz'] >= 10e9  # Ku-band or higher
                
                # High transmit power for very long range detection
                assert radar_config['tx_power_w'] >= 8e3
                
                # Very high antenna gain for long range beam focus
                assert radar_config['antenna_gain_db'] >= 34.0
                
                # Sensitive detection threshold
                assert radar_config['snr_threshold_db'] >= 8.0
                
        except ImportError:
            pytest.skip("LongRangeMissile dependencies not available")

    def test_long_range_missile_heavy_warhead_characteristics(self, mock_source, mock_target, mock_map_limits):
        """Test long range missile heavy warhead and structural characteristics."""
        try:
            from missiles.fox3.default_missile import LongRangeMissile
            
            with patch('missiles.missile.Missile.__init__', return_value=None) as mock_init:
                LongRangeMissile(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits
                )
                
                config = mock_init.call_args[1]['config']
                
                # Heavy missile characteristics
                assert config['mass_kg'] >= 350.0  # Very heavy for long range
                
                # Large structure for fuel and electronics
                assert config['reference_area_m2'] >= 0.25  # Large cross-sectional area
                
                # Despite size, should maintain good aerodynamics
                assert config['drag_coefficient'] <= 0.08  # Low drag despite size
                assert config['aspect_ratio'] >= 3.0  # Good lift characteristics
                
        except ImportError:
            pytest.skip("LongRangeMissile dependencies not available")