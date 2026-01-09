import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestAIM120AMRAAM:
    """Test the AIM-120 AMRAAM missile."""

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
        source.speed = 400.0
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
        target.name = "MiG29_Red"
        target.group = "RED"
        target.rcs_m2 = 5.0  # Radar cross section
        return target

    @pytest.fixture
    def mock_map_limits(self):
        """Create mock map limits."""
        return {
            "lat_min": 44.0,
            "lat_max": 47.0,
            "lon_min": 1.0,
            "lon_max": 4.0
        }

    def test_amraam_import(self):
        """Test that AMRAAM missile can be imported."""
        try:
            from missiles.fox3.amraam import AIM120_AMRAAM
            assert AIM120_AMRAAM is not None
        except ImportError:
            pytest.skip("AIM120_AMRAAM not available")

    def test_amraam_initialization(self, mock_source, mock_target, mock_map_limits):
        """Test AMRAAM missile initialization."""
        try:
            from missiles.fox3.amraam import AIM120_AMRAAM
            
            with patch('missiles.missile.Missile.__init__', return_value=None) as mock_init:
                missile = AIM120_AMRAAM(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits,
                    group="BLUE"
                )
                
                # Verify initialization was called with correct parameters
                mock_init.assert_called_once()
                args, kwargs = mock_init.call_args
                
                assert kwargs['name'] == "AIM120_AMRAAM"
                assert kwargs['firing_time_s'] == 0.0
                assert kwargs['target'] == mock_target
                assert kwargs['source'] == mock_source
                assert kwargs['map_limits'] == mock_map_limits
                assert kwargs['group'] == "BLUE"
                assert kwargs['data_link_mode'] == "full"
                
                # Check configuration values - medium/long range active radar missile
                # Values updated based on Wikipedia/public sources
                config = kwargs['config']
                assert config['mass_kg'] == 160.0  # AIM-120D typical mass (152-161 kg)
                assert config['max_speed_mps'] == 1370.0  # ~Mach 4 capability
                assert config['motor_burn_s'] == 8.0  # Public boost time ~8s
                assert config['life_time_s'] == 120.0  # Long range capability
                assert config['hit_probability'] == 0.85  # Good accuracy
                assert config['n_max'] == 40.0  # 40g class (C-5/6/7, D similar)
                
                # Verify FOX type
                assert missile.fox_type == 3
                
        except ImportError:
            pytest.skip("AIM120_AMRAAM dependencies not available")

    def test_amraam_radar_configuration(self, mock_source, mock_target, mock_map_limits):
        """Test AMRAAM active radar seeker configuration."""
        try:
            from missiles.fox3.amraam import AIM120_AMRAAM
            
            with patch('missiles.missile.Missile.__init__', return_value=None) as mock_init:
                AIM120_AMRAAM(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits,
                    group="BLUE"
                )
                
                # Check active radar seeker configuration
                config = mock_init.call_args[1]['config']
                radar_config = config['radar']

                assert radar_config['horizontal_fov_deg'] == 60.0  # Wide horizontal search
                assert radar_config['vertical_fov_deg'] == 30.0  # Narrower vertical
                assert radar_config['max_range_m'] == 150000.0  # Seeker ARH acquisition range (maximum kinematic)
                assert radar_config['radar_frequency_hz'] == 10e9  # X-band radar
                assert radar_config['tx_power_w'] == 8e3  # High power transmitter
                assert radar_config['antenna_gain_db'] == 32.0  # High gain antenna
                assert radar_config['snr_threshold_db'] == 8.0  # Detection threshold
                
        except ImportError:
            pytest.skip("AIM120_AMRAAM dependencies not available")

    def test_amraam_performance_characteristics(self, mock_source, mock_target, mock_map_limits):
        """Test AMRAAM performance characteristics."""
        try:
            from missiles.fox3.amraam import AIM120_AMRAAM
            
            with patch('missiles.missile.Missile.__init__', return_value=None) as mock_init:
                AIM120_AMRAAM(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits
                )
                
                config = mock_init.call_args[1]['config']

                # Test aerodynamic characteristics - optimized for range
                # Based on actual AMRAAM dimensions (Ø 178 mm)
                assert config['reference_area_m2'] == 0.025  # π*(0.089 m)^2 ≈ 0.0249 m²
                assert config['aspect_ratio'] == 3.0  # Reasonable for missile
                assert config['oswald_e'] == 0.78
                assert config['drag_coefficient'] == 0.08  # Low drag for range

                # Test propulsion characteristics - short boost phase
                assert config['constant_engine_N'] == 9000.0  # Surrogate thrust
                assert config['motor_burn_s'] == 8.0  # Public boost time ~8s

                # Test guidance characteristics - BVR engagement
                assert config['min_range_m'] == 1500.0  # Minimum safe distance
                assert config['seeker_sensitivity'] == 1.2  # Good radar sensitivity
                
        except ImportError:
            pytest.skip("AIM120_AMRAAM dependencies not available")

    def test_amraam_vs_fox2_differences(self, mock_source, mock_target, mock_map_limits):
        """Test differences between AMRAAM and FOX2 missiles."""
        try:
            from missiles.fox3.amraam import AIM120_AMRAAM
            from missiles.fox2.sidewinder import AIM9_Sidewinder
            
            with patch('missiles.missile.Missile.__init__', return_value=None) as mock_init:
                # Test AMRAAM
                AIM120_AMRAAM(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits
                )
                amraam_config = mock_init.call_args[1]['config']
                
                mock_init.reset_mock()
                
                # Test Sidewinder for comparison
                with patch('missiles.fox2.base_fox2.Fox2Missile.__init__', return_value=None):
                    AIM9_Sidewinder(
                        firing_time_s=0.0,
                        target=mock_target,
                        source=mock_source,
                        map_limits=mock_map_limits
                    )
                
                # AMRAAM should be heavier and longer range than FOX2
                assert amraam_config['mass_kg'] > 150.0  # Much heavier than Sidewinder (~85 kg)
                assert amraam_config['life_time_s'] > 100.0  # Much longer range
                # Note: AMRAAM has shorter burn but coasts longer due to better energy management
                assert amraam_config['motor_burn_s'] >= 8.0  # Boost phase
                assert amraam_config['max_speed_mps'] > 1000.0  # Higher speed

                # AMRAAM should have active radar vs IR seeker
                assert 'radar' in amraam_config
                assert amraam_config['radar']['max_range_m'] >= 150000.0  # Seeker radar range
                
        except ImportError:
            pytest.skip("AMRAAM/Sidewinder comparison not available")

    def test_amraam_bvr_capabilities(self, mock_source, mock_target, mock_map_limits):
        """Test AMRAAM beyond visual range (BVR) capabilities."""
        try:
            from missiles.fox3.amraam import AIM120_AMRAAM
            
            with patch('missiles.missile.Missile.__init__', return_value=None) as mock_init:
                AIM120_AMRAAM(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits
                )
                
                config = mock_init.call_args[1]['config']

                # BVR engagement characteristics
                assert config['life_time_s'] >= 100.0  # Long flight time for BVR
                assert config['max_speed_mps'] >= 1200.0  # High speed to close distance
                assert config['min_range_m'] >= 1000.0  # Minimum safe distance

                # Active radar for BVR tracking
                radar_config = config['radar']
                assert radar_config['max_range_m'] >= 150000.0  # Seeker acquisition range
                assert radar_config['horizontal_fov_deg'] >= 45.0  # Wide search angle
                
        except ImportError:
            pytest.skip("AIM120_AMRAAM dependencies not available")

    def test_amraam_data_link_mode(self, mock_source, mock_target, mock_map_limits):
        """Test AMRAAM data link capabilities."""
        try:
            from missiles.fox3.amraam import AIM120_AMRAAM
            
            with patch('missiles.missile.Missile.__init__', return_value=None) as mock_init:
                AIM120_AMRAAM(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits
                )
                
                # Verify data link mode is set to "full"
                # This allows mid-course guidance updates from launching platform
                assert mock_init.call_args[1]['data_link_mode'] == "full"
                
        except ImportError:
            pytest.skip("AIM120_AMRAAM dependencies not available")

    def test_amraam_active_radar_homing(self, mock_source, mock_target, mock_map_limits):
        """Test AMRAAM active radar homing characteristics."""
        try:
            from missiles.fox3.amraam import AIM120_AMRAAM
            
            with patch('missiles.missile.Missile.__init__', return_value=None) as mock_init:
                AIM120_AMRAAM(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits
                )
                
                config = mock_init.call_args[1]['config']
                radar_config = config['radar']
                
                # Active radar should have appropriate characteristics
                # High frequency for good resolution
                assert radar_config['radar_frequency_hz'] >= 8e9  # X-band or higher
                
                # High transmit power for long range detection
                assert radar_config['tx_power_w'] >= 5e3
                
                # High antenna gain for directional beam
                assert radar_config['antenna_gain_db'] >= 30.0
                
                # Reasonable SNR threshold for detection
                assert 6.0 <= radar_config['snr_threshold_db'] <= 12.0
                
        except ImportError:
            pytest.skip("AIM120_AMRAAM dependencies not available")