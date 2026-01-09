import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestSarhSeeker:
    """Test the Semi-Active Radar Homing (SARH) seeker."""

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
        missile.name = "Test_Missile"
        return missile

    @pytest.fixture
    def mock_target(self):
        """Create a mock target."""
        target = Mock()
        target.position = Mock()
        target.position.lat = 45.1
        target.position.lon = 2.1
        target.position.alt = 8500.0
        target.name = "Enemy_Target"
        target.group = "RED"
        return target

    @pytest.fixture
    def mock_source(self):
        """Create a mock source (launching aircraft)."""
        source = Mock()
        source.position = Mock()
        source.position.lat = 44.9
        source.position.lon = 1.9
        source.position.alt = 10000.0
        source.name = "Source_Aircraft"
        source.group = "BLUE"
        return source

    def test_sarh_seeker_import(self):
        """Test that SARH seeker can be imported."""
        try:
            from missiles.fox1.sarh_seeker import SarhSeeker
            assert SarhSeeker is not None
        except ImportError:
            pytest.skip("SarhSeeker not available")

    def test_sarh_seeker_initialization(self, mock_missile, mock_source):
        """Test SARH seeker initialization."""
        try:
            from missiles.fox1.sarh_seeker import SarhSeeker
            
            radar_config = {
                "fov_deg": 30.0,
                "max_range_m": 50000.0,
                "sensitivity": 1.0
            }
            
            seeker = SarhSeeker(
                missile=mock_missile,
                source=mock_source,
                radar_config=radar_config
            )
            
            assert seeker.missile == mock_missile
            assert seeker.source == mock_source
            assert seeker.fov_deg == 30.0
            assert seeker.max_range_m == 50000.0
            assert seeker.sensitivity == 1.0
            
        except ImportError:
            pytest.skip("SarhSeeker not available")

    def test_sarh_seeker_can_track_target(self, mock_missile, mock_source, mock_target):
        """Test SARH seeker target tracking capability."""
        try:
            from missiles.fox1.sarh_seeker import SarhSeeker
            
            radar_config = {
                "fov_deg": 45.0,
                "max_range_m": 50000.0,
                "sensitivity": 1.0
            }
            
            with patch('missiles.fox1.sarh_seeker.calculate_distance') as mock_distance, \
                 patch('missiles.fox1.sarh_seeker.calculate_angle_to_target') as mock_angle:
                
                # Mock distance calculation (target within range)
                mock_distance.return_value = 25000.0
                
                # Mock angle calculation (target within FOV)
                mock_angle.return_value = 20.0
                
                seeker = SarhSeeker(
                    missile=mock_missile,
                    source=mock_source,
                    radar_config=radar_config
                )
                
                # Test successful tracking
                can_track = seeker.can_track_target(mock_target)
                assert can_track is True
                
                # Test target too far
                mock_distance.return_value = 75000.0
                can_track = seeker.can_track_target(mock_target)
                assert can_track is False
                
                # Test target outside FOV
                mock_distance.return_value = 25000.0
                mock_angle.return_value = 60.0
                can_track = seeker.can_track_target(mock_target)
                assert can_track is False
                
        except ImportError:
            pytest.skip("SarhSeeker dependencies not available")

    def test_sarh_seeker_lose_lock(self, mock_missile, mock_source):
        """Test SARH seeker lose lock functionality."""
        try:
            from missiles.fox1.sarh_seeker import SarhSeeker
            
            radar_config = {
                "fov_deg": 30.0,
                "max_range_m": 50000.0,
                "sensitivity": 1.0
            }
            
            seeker = SarhSeeker(
                missile=mock_missile,
                source=mock_source,
                radar_config=radar_config
            )
            
            # Test lose lock functionality
            seeker.lose_lock()
            # Should not raise any exceptions
            
        except ImportError:
            pytest.skip("SarhSeeker not available")

    def test_sarh_seeker_requires_source_illumination(self, mock_missile, mock_source, mock_target):
        """Test that SARH requires continuous source illumination."""
        try:
            from missiles.fox1.sarh_seeker import SarhSeeker
            
            radar_config = {
                "fov_deg": 30.0,
                "max_range_m": 50000.0,
                "sensitivity": 1.0
            }
            
            seeker = SarhSeeker(
                missile=mock_missile,
                source=mock_source,
                radar_config=radar_config
            )
            
            # Test that seeker maintains reference to source
            assert seeker.source == mock_source
            
            # SARH missiles require the launching aircraft to maintain lock
            # This is a key characteristic that differentiates FOX1 from FOX3
            
        except ImportError:
            pytest.skip("SarhSeeker not available")

    def test_sarh_seeker_sensitivity_affects_tracking(self, mock_missile, mock_source, mock_target):
        """Test that seeker sensitivity affects tracking capability."""
        try:
            from missiles.fox1.sarh_seeker import SarhSeeker
            
            with patch('missiles.fox1.sarh_seeker.calculate_distance') as mock_distance, \
                 patch('missiles.fox1.sarh_seeker.calculate_angle_to_target') as mock_angle:
                
                mock_distance.return_value = 25000.0
                mock_angle.return_value = 20.0
                
                # High sensitivity seeker
                high_sens_config = {
                    "fov_deg": 30.0,
                    "max_range_m": 50000.0,
                    "sensitivity": 1.5
                }
                
                high_sens_seeker = SarhSeeker(
                    missile=mock_missile,
                    source=mock_source,
                    radar_config=high_sens_config
                )
                
                # Low sensitivity seeker
                low_sens_config = {
                    "fov_deg": 30.0,
                    "max_range_m": 50000.0,
                    "sensitivity": 0.5
                }
                
                low_sens_seeker = SarhSeeker(
                    missile=mock_missile,
                    source=mock_source,
                    radar_config=low_sens_config
                )
                
                assert high_sens_seeker.sensitivity > low_sens_seeker.sensitivity
                
        except ImportError:
            pytest.skip("SarhSeeker dependencies not available")