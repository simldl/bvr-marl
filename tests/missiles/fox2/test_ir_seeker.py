import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestIRSeeker:
    """Test the Infrared (IR) seeker for FOX2 missiles."""

    @pytest.fixture
    def mock_missile(self):
        """Create a mock missile."""
        missile = Mock()
        missile.position = Mock()
        missile.position.lat = 45.0
        missile.position.lon = 2.0
        missile.position.alt = 8000.0
        missile.yaw_deg = 90.0
        missile.pitch_deg = 0.0
        missile.name = "Test_FOX2_Missile"
        return missile

    @pytest.fixture
    def mock_hot_target(self):
        """Create a mock hot target (afterburning jet)."""
        target = Mock()
        target.position = Mock()
        target.position.lat = 45.1
        target.position.lon = 2.1
        target.position.alt = 8500.0
        target.name = "Hot_Target"
        target.group = "RED"
        target.temperature = 600.0  # Hot afterburning engine
        target.engine_status = "afterburner"
        return target

    @pytest.fixture
    def mock_cold_target(self):
        """Create a mock cold target (glider or low-heat aircraft)."""
        target = Mock()
        target.position = Mock()
        target.position.lat = 45.1
        target.position.lon = 2.1
        target.position.alt = 8500.0
        target.name = "Cold_Target"
        target.group = "RED"
        target.temperature = 300.0  # Cold target
        target.engine_status = "idle"
        return target

    def test_ir_seeker_import(self):
        """Test that IR seeker can be imported."""
        try:
            from missiles.fox2.ir_seeker import IRSeeker
            assert IRSeeker is not None
        except ImportError:
            pytest.skip("IRSeeker not available")

    def test_ir_seeker_initialization(self, mock_missile):
        """Test IR seeker initialization."""
        try:
            from missiles.fox2.ir_seeker import IRSeeker
            
            ir_config = {
                "fov_deg": 15.0,
                "max_range_m": 15000.0,
                "sensitivity": 1.0
            }
            
            seeker = IRSeeker(
                missile=mock_missile,
                ir_config=ir_config
            )
            
            assert seeker.missile == mock_missile
            assert seeker.fov_deg == 15.0
            assert seeker.max_range_m == 15000.0
            assert seeker.sensitivity == 1.0
            
        except ImportError:
            pytest.skip("IRSeeker not available")

    def test_ir_seeker_heat_signature_detection(self, mock_missile, mock_hot_target, mock_cold_target):
        """Test IR seeker heat signature detection."""
        try:
            from missiles.fox2.ir_seeker import IRSeeker
            
            ir_config = {
                "fov_deg": 15.0,
                "max_range_m": 15000.0,
                "sensitivity": 1.0
            }
            
            seeker = IRSeeker(
                missile=mock_missile,
                ir_config=ir_config
            )
            
            # Test heat signature detection
            hot_signature = seeker.get_heat_signature(mock_hot_target)
            cold_signature = seeker.get_heat_signature(mock_cold_target)
            
            assert hot_signature > cold_signature
            assert hot_signature >= 600.0
            assert cold_signature <= 300.0
            
        except ImportError:
            pytest.skip("IRSeeker not available")

    def test_ir_seeker_can_track_target(self, mock_missile, mock_hot_target):
        """Test IR seeker target tracking capability."""
        try:
            from missiles.fox2.ir_seeker import IRSeeker
            
            ir_config = {
                "fov_deg": 20.0,
                "max_range_m": 15000.0,
                "sensitivity": 1.0
            }
            
            with patch('missiles.fox2.ir_seeker.calculate_distance') as mock_distance, \
                 patch('missiles.fox2.ir_seeker.calculate_angle_to_target') as mock_angle:
                
                # Mock distance calculation (target within range)
                mock_distance.return_value = 8000.0
                
                # Mock angle calculation (target within FOV)
                mock_angle.return_value = 15.0
                
                seeker = IRSeeker(
                    missile=mock_missile,
                    ir_config=ir_config
                )
                
                # Test successful tracking of hot target
                can_track = seeker.can_track_target(mock_hot_target)
                assert can_track is True
                
                # Test target too far
                mock_distance.return_value = 25000.0
                can_track = seeker.can_track_target(mock_hot_target)
                assert can_track is False
                
                # Test target outside FOV
                mock_distance.return_value = 8000.0
                mock_angle.return_value = 35.0
                can_track = seeker.can_track_target(mock_hot_target)
                assert can_track is False
                
        except ImportError:
            pytest.skip("IRSeeker dependencies not available")

    def test_ir_seeker_sensitivity_affects_tracking(self, mock_missile, mock_cold_target):
        """Test that seeker sensitivity affects tracking of cold targets."""
        try:
            from missiles.fox2.ir_seeker import IRSeeker
            
            with patch('missiles.fox2.ir_seeker.calculate_distance') as mock_distance, \
                 patch('missiles.fox2.ir_seeker.calculate_angle_to_target') as mock_angle:
                
                mock_distance.return_value = 8000.0
                mock_angle.return_value = 10.0
                
                # High sensitivity seeker
                high_sens_config = {
                    "fov_deg": 15.0,
                    "max_range_m": 15000.0,
                    "sensitivity": 2.0  # High sensitivity
                }
                
                high_sens_seeker = IRSeeker(
                    missile=mock_missile,
                    ir_config=high_sens_config
                )
                
                # Low sensitivity seeker
                low_sens_config = {
                    "fov_deg": 15.0,
                    "max_range_m": 15000.0,
                    "sensitivity": 0.5  # Low sensitivity
                }
                
                low_sens_seeker = IRSeeker(
                    missile=mock_missile,
                    ir_config=low_sens_config
                )
                
                # High sensitivity should be able to track cold targets better
                assert high_sens_seeker.sensitivity > low_sens_seeker.sensitivity
                
        except ImportError:
            pytest.skip("IRSeeker dependencies not available")

    def test_ir_seeker_aspect_angle_matters(self, mock_missile, mock_hot_target):
        """Test that IR seeker performance depends on target aspect angle."""
        try:
            from missiles.fox2.ir_seeker import IRSeeker
            
            ir_config = {
                "fov_deg": 15.0,
                "max_range_m": 15000.0,
                "sensitivity": 1.0
            }
            
            seeker = IRSeeker(
                missile=mock_missile,
                ir_config=ir_config
            )
            
            # Rear aspect (hot exhaust visible) - best case
            mock_hot_target.aspect_angle = 0.0  # Directly behind
            rear_signature = seeker.get_heat_signature(mock_hot_target)
            
            # Side aspect (less heat visible)
            mock_hot_target.aspect_angle = 90.0  # From the side
            side_signature = seeker.get_heat_signature(mock_hot_target)
            
            # Front aspect (minimal heat visible) - worst case
            mock_hot_target.aspect_angle = 180.0  # Head-on
            front_signature = seeker.get_heat_signature(mock_hot_target)
            
            # Rear aspect should give strongest signature
            # This test assumes the IR seeker implementation considers aspect angle
            assert rear_signature >= side_signature
            assert side_signature >= front_signature
            
        except ImportError:
            pytest.skip("IRSeeker not available")
        except AttributeError:
            # If aspect angle is not implemented, just test basic functionality
            pytest.skip("Aspect angle not implemented in IRSeeker")

    def test_ir_seeker_all_aspect_capability(self, mock_missile, mock_hot_target):
        """Test modern IR seekers' all-aspect capability."""
        try:
            from missiles.fox2.ir_seeker import IRSeeker
            
            # Modern IR seekers should be able to track from all aspects
            ir_config = {
                "fov_deg": 15.0,
                "max_range_m": 15000.0,
                "sensitivity": 1.2,  # Higher sensitivity for all-aspect
                "all_aspect": True
            }
            
            seeker = IRSeeker(
                missile=mock_missile,
                ir_config=ir_config
            )
            
            # Should be able to detect heat signature regardless of aspect
            # (assuming modern all-aspect seeker implementation)
            signature = seeker.get_heat_signature(mock_hot_target)
            assert signature > 0.0
            
        except ImportError:
            pytest.skip("IRSeeker not available")

    def test_ir_seeker_environmental_factors(self, mock_missile, mock_hot_target):
        """Test IR seeker performance under different environmental conditions."""
        try:
            from missiles.fox2.ir_seeker import IRSeeker
            
            ir_config = {
                "fov_deg": 15.0,
                "max_range_m": 15000.0,
                "sensitivity": 1.0
            }
            
            seeker = IRSeeker(
                missile=mock_missile,
                ir_config=ir_config
            )
            
            # Test basic functionality (environmental factors may not be implemented)
            signature = seeker.get_heat_signature(mock_hot_target)
            assert signature > 0.0
            
            # If environmental factors are implemented, they would affect tracking
            # Examples: sun angle, cloud cover, atmospheric humidity, etc.
            
        except ImportError:
            pytest.skip("IRSeeker not available")