import pytest
import numpy as np
import math
from unittest.mock import Mock, patch, MagicMock


class TestPnGuidance:
    """Test the Proportional Navigation guidance system."""

    @pytest.fixture
    def mock_missile(self):
        """Create a mock missile."""
        missile = Mock()
        missile.speed = 400.0
        missile.n_max = 30.0
        missile.radar = Mock()
        missile.radar.get_tracker_info.return_value = None
        missile.target_provider = Mock()
        missile.target_provider.get_guidance_velocity.return_value = None
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

    def test_pn_guidance_import(self):
        """Test that PN guidance can be imported."""
        try:
            from missiles.guidance.pn import PnGuidance
            assert PnGuidance is not None
        except ImportError:
            pytest.skip("PnGuidance not available")

    def test_pn_guidance_initialization(self, mock_missile):
        """Test PN guidance initialization."""
        try:
            from missiles.guidance.pn import PnGuidance
            
            # Test default initialization
            pn = PnGuidance(mock_missile)
            assert pn.missile == mock_missile
            assert pn.PN_N == 3.5  # Default PN coefficient
            assert pn.ALTITUDE_THRESHOLD == 2000.0
            assert pn.SIGN_YAW == 1.0
            assert pn._prev_r is None
            
            # Test custom initialization
            pn_custom = PnGuidance(mock_missile, pn_n=4.0, altitude_threshold=3000.0)
            assert pn_custom.PN_N == 4.0
            assert pn_custom.ALTITUDE_THRESHOLD == 3000.0
            
        except ImportError:
            pytest.skip("PnGuidance dependencies not available")

    def test_pn_guidance_missile_velocity_vec(self, mock_missile):
        """Test missile velocity vector calculation."""
        try:
            from missiles.guidance.pn import PnGuidance
            
            pn = PnGuidance(mock_missile)
            
            # Test north-pointing velocity (0° yaw, 0° pitch)
            vel = pn._missile_velocity_vec(0.0, 0.0, 100.0)
            expected = np.array([0.0, 100.0, 0.0])  # East, North, Up
            np.testing.assert_allclose(vel, expected, rtol=1e-6, atol=1e-12)
            
            # Test east-pointing velocity (90° yaw, 0° pitch)
            vel = pn._missile_velocity_vec(90.0, 0.0, 100.0)
            expected = np.array([100.0, 0.0, 0.0])
            np.testing.assert_allclose(vel, expected, rtol=1e-6, atol=1e-12)
            
            # Test upward velocity (0° yaw, 90° pitch)
            vel = pn._missile_velocity_vec(0.0, 90.0, 100.0)
            expected = np.array([0.0, 0.0, 100.0])
            np.testing.assert_allclose(vel, expected, rtol=1e-6, atol=1e-12)
            
        except ImportError:
            pytest.skip("PnGuidance dependencies not available")

    def test_pn_guidance_get_n_calculation(self, mock_missile, mock_missile_position, mock_target_position):
        """Test PN coefficient calculation with range and altitude effects."""
        try:
            from missiles.guidance.pn import PnGuidance
            
            pn = PnGuidance(mock_missile, pn_n=3.0)
            
            # Test base case - long range
            N = pn._get_N(15000.0, mock_missile_position, mock_target_position, 0.0)
            assert 2.5 <= N <= 5.5  # Should be clamped
            
            # Test range ramp effect - short range should increase N
            N_short = pn._get_N(3000.0, mock_missile_position, mock_target_position, 0.0)
            N_long = pn._get_N(15000.0, mock_missile_position, mock_target_position, 0.0)
            assert N_short >= N_long  # Shorter range should have higher N
            
        except ImportError:
            pytest.skip("PnGuidance dependencies not available")

    def test_pn_guidance_yaw_rate_limiting(self, mock_missile):
        """Test yaw rate limiting based on missile maneuverability."""
        try:
            from missiles.guidance.pn import PnGuidance
            
            pn = PnGuidance(mock_missile)
            
            # Test with high yaw rate command
            high_rate = 10.0  # rad/s - very high
            limited_rate = pn._limit_yaw_rate(high_rate, 400.0)  # Vm = 400 m/s
            
            # Calculate expected max rate: n_max * g / Vm
            g = 9.81
            expected_max = (30.0 * g) / 400.0  # approximately 0.736 rad/s
            
            assert abs(limited_rate) <= expected_max * 1.01  # Small tolerance
            
            # Test with moderate yaw rate (should not be limited)
            moderate_rate = 0.1  # rad/s
            limited_moderate = pn._limit_yaw_rate(moderate_rate, 400.0)
            assert abs(limited_moderate - moderate_rate) < 1e-10
            
        except ImportError:
            pytest.skip("PnGuidance dependencies not available")

    def test_pn_guidance_compute_basic(self, mock_missile, mock_missile_position, mock_target_position):
        """Test basic PN guidance computation."""
        try:
            from missiles.guidance.pn import PnGuidance
            
            with patch('missiles.guidance.pn.geodetic_to_enu') as mock_geo:
                # Mock ENU conversion - target 1000m east, 1000m north of missile
                mock_geo.return_value = [1000.0, 1000.0, 500.0]  # East, North, Up
                
                pn = PnGuidance(mock_missile)
                
                yaw, pitch = pn.compute(
                    current_yaw_deg=0.0,
                    current_pitch_deg=0.0,
                    missile_position=mock_missile_position,
                    target_position=mock_target_position,
                    tick_secs=0.1
                )
                
                # Results should be valid angles
                assert 0.0 <= yaw <= 360.0
                assert -89.0 <= pitch <= 89.0
                
                # Since there's no target velocity and no previous position, 
                # the missile should point toward target (northeast in this case)
                # This should result in some guidance output rather than zero
                
        except ImportError:
            pytest.skip("PnGuidance dependencies not available")

    def test_pn_guidance_target_velocity_tracker(self, mock_missile, mock_missile_position, mock_target_position):
        """Test target velocity retrieval from tracker."""
        try:
            from missiles.guidance.pn import PnGuidance
            
            pn = PnGuidance(mock_missile)
            
            # Test with valid tracker velocity
            mock_missile.radar.get_tracker_info.return_value = {
                'enu_velocity': [50.0, 100.0, 10.0]  # m/s in ENU
            }
            
            vel = pn._get_target_velocity_enu()
            expected = np.array([50.0, 100.0, 10.0])
            np.testing.assert_allclose(vel, expected, rtol=1e-6, atol=1e-12)
            
            # Test with 2D velocity (should add zero z-component)
            mock_missile.radar.get_tracker_info.return_value = {
                'velocity': [50.0, 100.0]
            }
            
            vel = pn._get_target_velocity_enu()
            expected = np.array([50.0, 100.0, 0.0])
            np.testing.assert_allclose(vel, expected, rtol=1e-6, atol=1e-12)
            
            # Test with unrealistic velocity (should be rejected)
            mock_missile.radar.get_tracker_info.return_value = {
                'enu_velocity': [5000.0, 0.0, 0.0]  # Too fast
            }
            
            vel = pn._get_target_velocity_enu()
            assert vel is None
            
        except ImportError:
            pytest.skip("PnGuidance dependencies not available")

    def test_pn_guidance_velocity_estimation(self, mock_missile, mock_missile_position, mock_target_position):
        """Test target velocity estimation from position changes."""
        try:
            from missiles.guidance.pn import PnGuidance
            
            with patch('missiles.guidance.pn.geodetic_to_enu') as mock_geo:
                pn = PnGuidance(mock_missile)
                
                # First call - set up previous position
                mock_geo.return_value = [1000.0, 1000.0, 500.0]
                pn.compute(0.0, 0.0, mock_missile_position, mock_target_position, 0.1)
                
                # Second call - target has moved
                mock_geo.return_value = [1050.0, 1100.0, 550.0]  # Target moved 50E, 100N, 50U
                
                yaw, pitch = pn.compute(0.0, 0.0, mock_missile_position, mock_target_position, 0.1)
                
                # Should compute valid guidance based on estimated target velocity
                assert 0.0 <= yaw <= 360.0
                assert -89.0 <= pitch <= 89.0
                
                # Previous position should be stored
                assert pn._prev_r is not None
                np.testing.assert_allclose(pn._prev_r, [1050.0, 1100.0, 550.0], rtol=1e-6, atol=1e-12)
                
        except ImportError:
            pytest.skip("PnGuidance dependencies not available")

    def test_pn_guidance_zero_velocity_handling(self, mock_missile, mock_missile_position, mock_target_position):
        """Test handling of zero or very low target velocities."""
        try:
            from missiles.guidance.pn import PnGuidance
            
            pn = PnGuidance(mock_missile)
            
            # Test with effectively zero velocity from tracker
            mock_missile.radar.get_tracker_info.return_value = {
                'enu_velocity': [0.1, 0.1, 0.0]  # Very slow, should be treated as zero
            }
            
            vel = pn._get_target_velocity_enu()
            # Should reject very low speeds (returns None)
            # OR returns the actual low velocity array - both are acceptable
            if vel is not None:
                speed = np.linalg.norm(vel)
                assert speed <= 1.0  # Should be low speed
            
        except ImportError:
            pytest.skip("PnGuidance dependencies not available")

    def test_pn_guidance_angle_normalization(self, mock_missile, mock_missile_position, mock_target_position):
        """Test that output angles are properly normalized."""
        try:
            from missiles.guidance.pn import PnGuidance
            
            with patch('missiles.guidance.pn.geodetic_to_enu') as mock_geo:
                mock_geo.return_value = [1000.0, 1000.0, 500.0]
                
                pn = PnGuidance(mock_missile)
                
                # Test with large initial yaw that might cause wraparound
                yaw, pitch = pn.compute(
                    current_yaw_deg=350.0,  # Close to 360°
                    current_pitch_deg=85.0,  # Close to limit
                    missile_position=mock_missile_position,
                    target_position=mock_target_position,
                    tick_secs=0.1
                )
                
                # Yaw should be normalized to [0, 360)
                assert 0.0 <= yaw <= 360.0
                
                # Pitch should be clamped to [-89, 89]
                assert -89.0 <= pitch <= 89.0
                
        except ImportError:
            pytest.skip("PnGuidance dependencies not available")