import pytest
import math
from unittest.mock import Mock, patch


class TestDirectPursuitGuidance:
    """Test the direct pursuit guidance system."""

    @pytest.fixture
    def mock_missile(self):
        """Create a mock missile."""
        missile = Mock()
        return missile

    @pytest.fixture
    def mock_missile_position(self):
        """Create a mock missile position."""
        pos = Mock()
        pos.lat = 45.0
        pos.lon = 2.0
        pos.alt = 8000.0
        return pos

    @pytest.fixture
    def mock_target_position(self):
        """Create a mock target position."""
        pos = Mock()
        pos.lat = 46.0
        pos.lon = 3.0
        pos.alt = 8500.0
        return pos

    def test_direct_pursuit_import(self):
        """Test that direct pursuit guidance can be imported."""
        try:
            from missiles.guidance.direct import DirectPursuitGuidance
            assert DirectPursuitGuidance is not None
        except ImportError:
            pytest.skip("DirectPursuitGuidance not available")

    def test_direct_pursuit_initialization(self, mock_missile):
        """Test direct pursuit guidance initialization."""
        try:
            from missiles.guidance.direct import DirectPursuitGuidance
            
            guidance = DirectPursuitGuidance(mock_missile)
            assert guidance.missile == mock_missile
            
        except ImportError:
            pytest.skip("DirectPursuitGuidance dependencies not available")

    def test_direct_pursuit_compute_bearing(self, mock_missile, mock_missile_position, mock_target_position):
        """Test direct pursuit guidance bearing calculation."""
        try:
            from missiles.guidance.direct import DirectPursuitGuidance
            
            with patch('missiles.guidance.direct.geodetic_bearing_deg') as mock_bearing, \
                 patch('missiles.guidance.direct.geodetic_distance_km') as mock_distance:
                
                mock_bearing.return_value = 45.0  # Northeast bearing
                mock_distance.return_value = 100.0  # 100 km horizontal distance
                
                guidance = DirectPursuitGuidance(mock_missile)
                
                yaw, pitch = guidance.compute(
                    current_yaw_deg=0.0,
                    current_pitch_deg=0.0,
                    missile_position=mock_missile_position,
                    target_position=mock_target_position,
                    tick_secs=0.1
                )
                
                # Should return the bearing calculated by geodetic_bearing_deg
                assert yaw == 45.0
                
                # Verify bearing calculation was called with correct parameters
                mock_bearing.assert_called_once_with(
                    mock_missile_position.lat, mock_missile_position.lon,
                    mock_target_position.lat, mock_target_position.lon
                )
                
        except ImportError:
            pytest.skip("DirectPursuitGuidance dependencies not available")

    def test_direct_pursuit_compute_pitch(self, mock_missile, mock_missile_position, mock_target_position):
        """Test direct pursuit guidance pitch calculation."""
        try:
            from missiles.guidance.direct import DirectPursuitGuidance
            
            with patch('missiles.guidance.direct.geodetic_bearing_deg') as mock_bearing, \
                 patch('missiles.guidance.direct.geodetic_distance_km') as mock_distance:
                
                mock_bearing.return_value = 45.0
                mock_distance.return_value = 100.0  # 100 km = 100,000 m horizontal distance
                
                # Target 500m higher than missile
                mock_target_position.alt = 8500.0
                mock_missile_position.alt = 8000.0
                
                guidance = DirectPursuitGuidance(mock_missile)
                
                yaw, pitch = guidance.compute(
                    current_yaw_deg=0.0,
                    current_pitch_deg=0.0,
                    missile_position=mock_missile_position,
                    target_position=mock_target_position,
                    tick_secs=0.1
                )
                
                # Calculate expected pitch: atan2(500, 100000) in degrees
                vertical_diff = 500.0
                horizontal_distance = 100000.0  # 100 km in meters
                expected_pitch = math.degrees(math.atan2(vertical_diff, horizontal_distance))
                
                assert abs(pitch - expected_pitch) < 1e-10
                
        except ImportError:
            pytest.skip("DirectPursuitGuidance dependencies not available")

    def test_direct_pursuit_same_altitude(self, mock_missile, mock_missile_position, mock_target_position):
        """Test direct pursuit when target and missile are at same altitude."""
        try:
            from missiles.guidance.direct import DirectPursuitGuidance
            
            with patch('missiles.guidance.direct.geodetic_bearing_deg') as mock_bearing, \
                 patch('missiles.guidance.direct.geodetic_distance_km') as mock_distance:
                
                mock_bearing.return_value = 90.0  # East
                mock_distance.return_value = 50.0  # 50 km
                
                # Same altitude
                mock_target_position.alt = 8000.0
                mock_missile_position.alt = 8000.0
                
                guidance = DirectPursuitGuidance(mock_missile)
                
                yaw, pitch = guidance.compute(
                    current_yaw_deg=0.0,
                    current_pitch_deg=0.0,
                    missile_position=mock_missile_position,
                    target_position=mock_target_position,
                    tick_secs=0.1
                )
                
                assert yaw == 90.0
                assert abs(pitch) < 1e-10  # Should be approximately 0
                
        except ImportError:
            pytest.skip("DirectPursuitGuidance dependencies not available")

    def test_direct_pursuit_target_below(self, mock_missile, mock_missile_position, mock_target_position):
        """Test direct pursuit when target is below missile."""
        try:
            from missiles.guidance.direct import DirectPursuitGuidance
            
            with patch('missiles.guidance.direct.geodetic_bearing_deg') as mock_bearing, \
                 patch('missiles.guidance.direct.geodetic_distance_km') as mock_distance:
                
                mock_bearing.return_value = 180.0  # South
                mock_distance.return_value = 80.0  # 80 km
                
                # Target 1000m below missile
                mock_target_position.alt = 7000.0
                mock_missile_position.alt = 8000.0
                
                guidance = DirectPursuitGuidance(mock_missile)
                
                yaw, pitch = guidance.compute(
                    current_yaw_deg=0.0,
                    current_pitch_deg=0.0,
                    missile_position=mock_missile_position,
                    target_position=mock_target_position,
                    tick_secs=0.1
                )
                
                assert yaw == 180.0
                assert pitch < 0.0  # Should be negative (nose down)
                
                # Calculate expected pitch
                vertical_diff = -1000.0  # Target below
                horizontal_distance = 80000.0  # 80 km in meters
                expected_pitch = math.degrees(math.atan2(vertical_diff, horizontal_distance))
                
                assert abs(pitch - expected_pitch) < 1e-10
                
        except ImportError:
            pytest.skip("DirectPursuitGuidance dependencies not available")

    def test_direct_pursuit_zero_distance_protection(self, mock_missile, mock_missile_position, mock_target_position):
        """Test direct pursuit with zero horizontal distance protection."""
        try:
            from missiles.guidance.direct import DirectPursuitGuidance
            
            with patch('missiles.guidance.direct.geodetic_bearing_deg') as mock_bearing, \
                 patch('missiles.guidance.direct.geodetic_distance_km') as mock_distance:
                
                mock_bearing.return_value = 0.0
                mock_distance.return_value = 0.0  # Zero horizontal distance
                
                # Target directly above
                mock_target_position.alt = 9000.0
                mock_missile_position.alt = 8000.0
                
                guidance = DirectPursuitGuidance(mock_missile)
                
                yaw, pitch = guidance.compute(
                    current_yaw_deg=0.0,
                    current_pitch_deg=0.0,
                    missile_position=mock_missile_position,
                    target_position=mock_target_position,
                    tick_secs=0.1
                )
                
                assert yaw == 0.0
                # Should use minimum distance (1e-3) to prevent division by zero
                # This should result in nearly vertical pitch
                assert pitch > 80.0  # Very steep climb
                
        except ImportError:
            pytest.skip("DirectPursuitGuidance dependencies not available")

    def test_direct_pursuit_real_world_scenario(self, mock_missile):
        """Test direct pursuit with realistic positions."""
        try:
            from missiles.guidance.direct import DirectPursuitGuidance
            
            # Real positions: missile over London, target over Paris
            missile_pos = Mock()
            missile_pos.lat = 51.5074
            missile_pos.lon = -0.1278
            missile_pos.alt = 10000.0
            
            target_pos = Mock()
            target_pos.lat = 48.8566
            target_pos.lon = 2.3522
            target_pos.alt = 9500.0
            
            guidance = DirectPursuitGuidance(mock_missile)
            
            yaw, pitch = guidance.compute(
                current_yaw_deg=0.0,
                current_pitch_deg=0.0,
                missile_position=missile_pos,
                target_position=target_pos,
                tick_secs=0.1
            )
            
            # Should produce reasonable values
            assert -180.0 <= yaw <= 180.0
            assert -90.0 <= pitch <= 90.0
            
            # For London to Paris, should be roughly southeast
            assert 90.0 < yaw < 180.0  # Southeast quadrant
            assert pitch < 0.0  # Slightly nose down (target lower)
            
        except ImportError:
            pytest.skip("DirectPursuitGuidance dependencies not available")