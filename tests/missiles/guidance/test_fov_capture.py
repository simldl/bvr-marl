from unittest.mock import Mock, patch

import pytest


class TestFovCaptureGuidance:
    """Test the FOV capture guidance system."""

    @pytest.fixture
    def mock_missile(self):
        """Create a mock missile."""
        return Mock()

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

    def test_fov_capture_import(self):
        """Test that FOV capture guidance can be imported."""
        from air_to_air_rl.missiles.guidance.fov_capture import FovCaptureGuidance

        assert FovCaptureGuidance is not None

    def test_fov_capture_initialization(self, mock_missile):
        """Test FOV capture guidance initialization."""
        from air_to_air_rl.missiles.guidance.fov_capture import FovCaptureGuidance

        guidance = FovCaptureGuidance(mock_missile)
        assert guidance.missile == mock_missile
        assert guidance.YAW_GAIN == 0.4
        assert guidance.PITCH_GAIN == 0.4

    def test_fov_capture_compute(self, mock_missile, mock_missile_position, mock_target_position):
        """Test FOV capture guidance computation."""
        from air_to_air_rl.missiles.guidance.fov_capture import FovCaptureGuidance

        with patch("air_to_air_rl.missiles.guidance.fov_capture._angles_dist") as mock_angles:
            # Mock angle calculations
            azimuth = 20.0  # degrees
            elevation = 10.0  # degrees
            distance = 5000.0  # meters
            mock_angles.return_value = (azimuth, elevation, distance)

            guidance = FovCaptureGuidance(mock_missile)

            current_yaw = 45.0
            current_pitch = 5.0

            yaw, pitch = guidance.compute(
                current_yaw_deg=current_yaw,
                current_pitch_deg=current_pitch,
                missile_position=mock_missile_position,
                target_position=mock_target_position,
                tick_secs=0.1,
            )

            # Expected: current + gain * angle error
            expected_yaw = current_yaw + 0.4 * azimuth  # 45.0 + 0.4 * 20.0 = 53.0
            expected_pitch = current_pitch + 0.4 * elevation  # 5.0 + 0.4 * 10.0 = 9.0

            assert abs(yaw - expected_yaw) < 1e-10
            assert abs(pitch - expected_pitch) < 1e-10

            # Verify _angles_dist was called with correct parameters
            mock_angles.assert_called_once_with(
                mock_missile_position, current_yaw, current_pitch, mock_target_position
            )

    def test_fov_capture_negative_angles(
        self, mock_missile, mock_missile_position, mock_target_position
    ):
        """Test FOV capture with negative angle errors."""
        from air_to_air_rl.missiles.guidance.fov_capture import FovCaptureGuidance

        with patch("air_to_air_rl.missiles.guidance.fov_capture._angles_dist") as mock_angles:
            # Mock negative angle errors (target behind current pointing)
            azimuth = -15.0  # degrees
            elevation = -8.0  # degrees
            distance = 3000.0  # meters
            mock_angles.return_value = (azimuth, elevation, distance)

            guidance = FovCaptureGuidance(mock_missile)

            current_yaw = 90.0
            current_pitch = 10.0

            yaw, pitch = guidance.compute(
                current_yaw_deg=current_yaw,
                current_pitch_deg=current_pitch,
                missile_position=mock_missile_position,
                target_position=mock_target_position,
                tick_secs=0.1,
            )

            # Expected: current + gain * angle error
            expected_yaw = current_yaw + 0.4 * azimuth  # 90.0 + 0.4 * (-15.0) = 84.0
            expected_pitch = current_pitch + 0.4 * elevation  # 10.0 + 0.4 * (-8.0) = 6.8

            assert abs(yaw - expected_yaw) < 1e-10
            assert abs(pitch - expected_pitch) < 1e-10

    def test_fov_capture_zero_angles(
        self, mock_missile, mock_missile_position, mock_target_position
    ):
        """Test FOV capture when target is already in center of FOV."""
        from air_to_air_rl.missiles.guidance.fov_capture import FovCaptureGuidance

        with patch("air_to_air_rl.missiles.guidance.fov_capture._angles_dist") as mock_angles:
            # Mock zero angle errors (target already centered)
            azimuth = 0.0
            elevation = 0.0
            distance = 2000.0
            mock_angles.return_value = (azimuth, elevation, distance)

            guidance = FovCaptureGuidance(mock_missile)

            current_yaw = 180.0
            current_pitch = -5.0

            yaw, pitch = guidance.compute(
                current_yaw_deg=current_yaw,
                current_pitch_deg=current_pitch,
                missile_position=mock_missile_position,
                target_position=mock_target_position,
                tick_secs=0.1,
            )

            # Should maintain current angles when no correction needed
            assert yaw == current_yaw
            assert pitch == current_pitch

    def test_fov_capture_large_angles(
        self, mock_missile, mock_missile_position, mock_target_position
    ):
        """Test FOV capture with large angle errors."""
        from air_to_air_rl.missiles.guidance.fov_capture import FovCaptureGuidance

        with patch("air_to_air_rl.missiles.guidance.fov_capture._angles_dist") as mock_angles:
            # Mock large angle errors
            azimuth = 60.0  # degrees
            elevation = 30.0  # degrees
            distance = 8000.0  # meters
            mock_angles.return_value = (azimuth, elevation, distance)

            guidance = FovCaptureGuidance(mock_missile)

            current_yaw = 0.0
            current_pitch = 0.0

            yaw, pitch = guidance.compute(
                current_yaw_deg=current_yaw,
                current_pitch_deg=current_pitch,
                missile_position=mock_missile_position,
                target_position=mock_target_position,
                tick_secs=0.1,
            )

            # Expected corrections with 0.4 gain
            expected_yaw = 0.0 + 0.4 * 60.0  # 24.0
            expected_pitch = 0.0 + 0.4 * 30.0  # 12.0

            assert abs(yaw - expected_yaw) < 1e-10
            assert abs(pitch - expected_pitch) < 1e-10

    def test_fov_capture_proportional_correction(
        self, mock_missile, mock_missile_position, mock_target_position
    ):
        """Test that FOV capture applies proportional correction."""
        from air_to_air_rl.missiles.guidance.fov_capture import FovCaptureGuidance

        with patch("air_to_air_rl.missiles.guidance.fov_capture._angles_dist") as mock_angles:
            guidance = FovCaptureGuidance(mock_missile)

            current_yaw = 45.0
            current_pitch = 10.0

            # Test with small angle error
            mock_angles.return_value = (5.0, 2.0, 1000.0)
            yaw1, pitch1 = guidance.compute(
                current_yaw, current_pitch, mock_missile_position, mock_target_position, 0.1
            )

            # Test with larger angle error
            mock_angles.return_value = (10.0, 4.0, 1000.0)
            yaw2, pitch2 = guidance.compute(
                current_yaw, current_pitch, mock_missile_position, mock_target_position, 0.1
            )

            # Corrections should be proportional to angle errors
            correction1_yaw = yaw1 - current_yaw
            correction2_yaw = yaw2 - current_yaw
            correction1_pitch = pitch1 - current_pitch
            correction2_pitch = pitch2 - current_pitch

            # Double the angle error should give double the correction
            assert abs(correction2_yaw - 2 * correction1_yaw) < 1e-10
            assert abs(correction2_pitch - 2 * correction1_pitch) < 1e-10

    def test_fov_capture_inheritance(self, mock_missile):
        """Test that FOV capture inherits from BaseGuidanceMode."""
        from air_to_air_rl.missiles.guidance.base import BaseGuidanceMode
        from air_to_air_rl.missiles.guidance.fov_capture import FovCaptureGuidance

        guidance = FovCaptureGuidance(mock_missile)
        assert isinstance(guidance, BaseGuidanceMode)
