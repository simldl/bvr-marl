from unittest.mock import Mock, patch

import pytest


class TestLoftGuidance:
    """Test the loft guidance system."""

    @pytest.fixture
    def mock_missile(self):
        """Create a mock missile."""
        missile = Mock()
        missile.position = Mock()
        missile.position.lat = 45.0
        missile.position.lon = 2.0
        missile.position.alt = 5000.0
        missile.phase_manager = Mock()
        missile.phase_manager.current_phase = "boost"
        return missile

    @pytest.fixture
    def mock_target_position(self):
        """Create a mock target position."""
        pos = Mock()
        pos.lat = 46.0
        pos.lon = 3.0
        pos.alt = 9000.0
        return pos

    def test_loft_guidance_import(self):
        """Test that loft guidance can be imported."""
        from bvr_marl_core.missiles.guidance.loft import LoftGuidance

        assert LoftGuidance is not None

    def test_loft_guidance_initialization(self, mock_missile):
        """Test loft guidance initialization."""
        from bvr_marl_core.missiles.guidance.loft import LoftGuidance

        guidance = LoftGuidance(mock_missile)
        assert guidance.missile == mock_missile
        assert guidance.MIN_LOFT_ALT == 15000.0
        assert guidance.MAX_LOFT_ALT == 30000.0
        assert guidance.FORCED_PITCH_CLIMB == 75.0

    def test_loft_guidance_forced_climb(self, mock_missile, mock_target_position):
        """Test loft guidance forced climb behavior."""
        from bvr_marl_core.missiles.guidance.loft import LoftGuidance

        with patch("bvr_marl_core.missiles.guidance.loft.geodetic_bearing_deg") as mock_bearing:
            mock_bearing.return_value = 45.0
            # Missile well below target altitude, in boost phase
            mock_missile.position.alt = 5000.0
            mock_missile.phase_manager.current_phase = "boost"
            # Static method call: compute(missile, target_pos, min_loft_range_m, alt_guard_m)
            result = LoftGuidance(mock_missile).compute(mock_missile, mock_target_position)
            # Should return loft commands: (yaw_geo, pitch)
            assert result is not None
            yaw, pitch = result
            assert yaw == 45.0
            assert pitch == 75.0  # FORCED_PITCH_CLIMB
            mock_bearing.assert_called_once_with(
                mock_missile.position.lat,
                mock_missile.position.lon,
                mock_target_position.lat,
                mock_target_position.lon,
            )

    def test_loft_guidance_near_target_altitude(self, mock_missile, mock_target_position):
        """Test loft guidance when near target altitude (should return None)."""
        from bvr_marl_core.missiles.guidance.loft import LoftGuidance

        # Missile significantly above target altitude (fails altitude guard check)
        # Need: missile.alt > (target.alt + alt_guard_m) to trigger guard
        mock_missile.position.alt = 11000.0  # 2000m above target
        mock_target_position.alt = 9000.0
        mock_missile.phase_manager.current_phase = "boost"
        # Should return None due to altitude guard (11000 > 9000 + 1500)
        result = LoftGuidance(mock_missile).compute(
            mock_missile, mock_target_position, alt_guard_m=1500.0
        )
        assert result is None  # No loft when already significantly above target

    def test_loft_guidance_above_target_altitude(self, mock_missile, mock_target_position):
        """Test loft guidance when above target altitude."""
        from bvr_marl_core.missiles.guidance.loft import LoftGuidance

        # Missile above target altitude
        mock_missile.position.alt = 12000.0
        mock_target_position.alt = 9000.0
        mock_missile.phase_manager.current_phase = "boost"
        # Should return None when altitude check fails
        result = LoftGuidance(mock_missile).compute(mock_missile, mock_target_position)
        assert result is None

    def test_loft_guidance_default_target_altitude(self, mock_missile, mock_target_position):
        """Test loft guidance with normal case."""
        from bvr_marl_core.missiles.guidance.loft import LoftGuidance

        with patch("bvr_marl_core.missiles.guidance.loft.geodetic_bearing_deg") as mock_bearing:
            mock_bearing.return_value = 0.0
            # Missile below target, in boost phase, sufficient range
            mock_missile.position.alt = 8000.0
            mock_target_position.alt = 10000.0
            mock_missile.phase_manager.current_phase = "boost"
            result = LoftGuidance(mock_missile).compute(
                mock_missile, mock_target_position, min_loft_range_m=1000.0
            )
            assert result is not None
            yaw, pitch = result
            assert yaw == 0.0
            assert pitch == 75.0  # FORCED_PITCH_CLIMB

    def test_loft_guidance_altitude_limits(self, mock_missile, mock_target_position):
        """Test loft guidance with various altitude configurations."""
        from bvr_marl_core.missiles.guidance.loft import LoftGuidance

        with patch("bvr_marl_core.missiles.guidance.loft.geodetic_bearing_deg") as mock_bearing:
            mock_bearing.return_value = 270.0
            # Low altitude missile should loft
            mock_missile.position.alt = 5000.0
            mock_target_position.alt = 15000.0
            mock_missile.phase_manager.current_phase = "boost"
            result = LoftGuidance(mock_missile).compute(mock_missile, mock_target_position)
            assert result is not None
            yaw, pitch = result
            assert yaw == 270.0
            assert pitch == 75.0

    def test_loft_guidance_threshold_boundary(self, mock_missile, mock_target_position):
        """Test loft guidance with range threshold."""
        from bvr_marl_core.missiles.guidance.loft import LoftGuidance

        with patch("bvr_marl_core.missiles.guidance.loft.geodetic_bearing_deg") as mock_bearing:
            mock_bearing.return_value = 180.0
            # Set up missile below target, in boost phase
            mock_missile.position.alt = 8000.0
            mock_target_position.alt = 10000.0
            mock_missile.phase_manager.current_phase = "boost"
            # Test with sufficient range
            result = LoftGuidance(mock_missile).compute(
                mock_missile, mock_target_position, min_loft_range_m=1000.0
            )
            assert result is not None
            yaw, pitch = result
            assert yaw == 180.0
            assert pitch == 75.0  # FORCED_PITCH_CLIMB

    def test_loft_guidance_not_in_boost_phase(self, mock_missile, mock_target_position):
        """Test that loft only works in boost phase."""
        from bvr_marl_core.missiles.guidance.loft import LoftGuidance

        # Missile not in boost phase
        mock_missile.position.alt = 5000.0
        mock_target_position.alt = 10000.0
        mock_missile.phase_manager.current_phase = "middle"  # Not boost
        result = LoftGuidance(mock_missile).compute(mock_missile, mock_target_position)
        # Should return None when not in boost phase
        assert result is None

    def test_loft_guidance_range_too_close(self, mock_missile, mock_target_position):
        """Test that loft doesn't activate when too close to target."""
        from bvr_marl_core.missiles.guidance.loft import LoftGuidance

        # Missile close to target (< min_loft_range)
        mock_missile.position.lat = 45.0
        mock_missile.position.lon = 2.0
        mock_missile.position.alt = 8000.0
        mock_target_position.lat = 45.001  # Very close in lat
        mock_target_position.lon = 2.001  # Very close in lon
        mock_target_position.alt = 9000.0
        mock_missile.phase_manager.current_phase = "boost"
        # Require large minimum range
        result = LoftGuidance(mock_missile).compute(
            mock_missile, mock_target_position, min_loft_range_m=50000.0
        )
        # Should return None when range check fails
        assert result is None
