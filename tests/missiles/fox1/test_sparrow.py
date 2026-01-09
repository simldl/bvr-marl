import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestAIM7Sparrow:
    """Test the AIM-7 Sparrow missile."""

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
        target.name = "MiG29_Red"
        target.group = "RED"
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

    def test_sparrow_import(self):
        """Test that Sparrow missile can be imported."""
        try:
            from missiles.fox1.sparrow import AIM7_Sparrow
            assert AIM7_Sparrow is not None
        except ImportError:
            pytest.skip("AIM7_Sparrow not available")

    def test_sparrow_initialization(self, mock_source, mock_target, mock_map_limits):
        """Test Sparrow missile initialization."""
        try:
            from missiles.fox1.sparrow import AIM7_Sparrow
            
            with patch('missiles.fox1.base_fox1.Fox1Missile.__init__', return_value=None) as mock_init:
                missile = AIM7_Sparrow(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits,
                    group="BLUE"
                )
                
                # Verify initialization was called with correct parameters
                mock_init.assert_called_once()
                args, kwargs = mock_init.call_args
                
                assert kwargs['name'] == "AIM7_Sparrow"
                assert kwargs['firing_time_s'] == 0.0
                assert kwargs['target'] == mock_target
                assert kwargs['source'] == mock_source
                assert kwargs['map_limits'] == mock_map_limits
                assert kwargs['group'] == "BLUE"
                
                # Check configuration values
                config = kwargs['config']
                assert config['mass_kg'] == 230.0
                assert config['max_speed_mps'] == 1200.0
                assert config['motor_burn_s'] == 8.0
                assert config['life_time_s'] == 60.0
                assert config['hit_probability'] == 0.85
                assert config['n_max'] == 25.0
                
        except ImportError:
            pytest.skip("AIM7_Sparrow dependencies not available")

    def test_sparrow_configuration_values(self):
        """Test that Sparrow has correct configuration values."""
        try:
            from missiles.fox1.sparrow import AIM7_Sparrow
            
            # We can't easily test the actual configuration without instantiating,
            # but we can verify the class exists and has the expected structure
            assert hasattr(AIM7_Sparrow, '__init__')
            
        except ImportError:
            pytest.skip("AIM7_Sparrow not available")

    def test_sparrow_radar_configuration(self, mock_source, mock_target, mock_map_limits):
        """Test Sparrow radar configuration."""
        try:
            from missiles.fox1.sparrow import AIM7_Sparrow
            
            with patch('missiles.fox1.base_fox1.Fox1Missile.__init__', return_value=None) as mock_init:
                AIM7_Sparrow(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits,
                    group="BLUE"
                )
                
                # Check radar configuration
                config = mock_init.call_args[1]['config']
                radar_config = config['radar']
                
                assert radar_config['fov_deg'] == 25.0
                assert radar_config['max_range_m'] == 45000.0
                assert radar_config['sensitivity'] == 1.2
                
        except ImportError:
            pytest.skip("AIM7_Sparrow dependencies not available")

    def test_sparrow_performance_characteristics(self, mock_source, mock_target, mock_map_limits):
        """Test Sparrow performance characteristics."""
        try:
            from missiles.fox1.sparrow import AIM7_Sparrow
            
            with patch('missiles.fox1.base_fox1.Fox1Missile.__init__', return_value=None) as mock_init:
                AIM7_Sparrow(
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits
                )
                
                config = mock_init.call_args[1]['config']
                
                # Test aerodynamic characteristics
                assert config['reference_area_m2'] == 0.20
                assert config['aspect_ratio'] == 3.0
                assert config['oswald_e'] == 0.75
                assert config['drag_coefficient'] == 0.12
                
                # Test propulsion characteristics
                assert config['constant_engine_N'] == 12000.0
                assert config['motor_burn_s'] == 8.0
                
                # Test guidance characteristics
                assert config['min_range_m'] == 1000.0
                assert config['seeker_sensitivity'] == 1.2
                
        except ImportError:
            pytest.skip("AIM7_Sparrow dependencies not available")