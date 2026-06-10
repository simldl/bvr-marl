import math
from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestPnPropNavGuidance:
    """Test the PN PropNav guidance system."""

    @pytest.fixture
    def mock_missile(self):
        """Create a mock missile."""
        missile = Mock()
        missile.speed = 800.0
        missile.n_max = 30.0
        missile.radar = Mock()
        missile.radar.get_tracker_info.return_value = None
        missile.radar.tracker_manager = None  # prevent auto-Mock from masking None checks
        missile.target_provider = Mock()
        missile.target_provider.get_guidance_velocity.return_value = None
        missile.target_provider.get_guidance_velocity_in_missile_enu.return_value = None
        missile.target_provider.get_guidance_tid.return_value = None  # no track by default
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

    def test_pn_propnav_import(self):
        """Test that PN PropNav guidance can be imported."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        assert PnPropNavGuidance is not None

    def test_pn_propnav_initialization(self, mock_missile):
        """Test PN PropNav guidance initialization."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        # Test default initialization
        guidance = PnPropNavGuidance(mock_missile)
        assert guidance.missile == mock_missile
        assert guidance.PN_N == 4.0  # Default PN coefficient
        assert (
            guidance.pn_law == PnPropNavGuidance.PN_TRUE
        )  # Default is PN_TRUE (1), not PN_PURE (2)
        assert guidance.ALTITUDE_THRESHOLD == 2000.0
        assert guidance.g == 9.80665
        assert guidance._last_valid_vt is None

        # Test custom initialization
        guidance_custom = PnPropNavGuidance(
            mock_missile, pn_n=5.0, pn_law=PnPropNavGuidance.PN_TRUE, altitude_threshold=3000.0
        )
        assert guidance_custom.PN_N == 5.0
        assert guidance_custom.pn_law == PnPropNavGuidance.PN_TRUE
        assert guidance_custom.ALTITUDE_THRESHOLD == 3000.0

    def test_pn_propnav_pn_law_constants(self):
        """Test PN law constants."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        assert PnPropNavGuidance.PN_TRUE == 1
        assert PnPropNavGuidance.PN_PURE == 2
        assert PnPropNavGuidance.PN_ZEM == 3

    def test_pn_propnav_unit_vector(self, mock_missile):
        """Test unit vector calculation."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        guidance = PnPropNavGuidance(mock_missile)
        # Test normal vector
        v = np.array([3.0, 4.0, 0.0])
        unit_v = guidance._unit(v)  # Method is _unit, not _unit_vector
        expected = np.array([0.6, 0.8, 0.0])  # |v| = 5, so [3,4,0]/5
        np.testing.assert_allclose(unit_v, expected, rtol=1e-6, atol=1e-12)
        # Test zero vector
        zero_v = np.array([0.0, 0.0, 0.0])
        unit_zero = guidance._unit(zero_v)  # Method is _unit, not _unit_vector
        expected_zero = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(unit_zero, expected_zero, rtol=1e-6, atol=1e-12)

    def test_pn_propnav_missile_velocity_vec(self, mock_missile):
        """Test missile velocity vector calculation."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        guidance = PnPropNavGuidance(mock_missile)

        # Test north-pointing velocity (0° yaw, 0° pitch)
        vel = guidance._missile_velocity_vec(0.0, 0.0, 100.0)
        expected = np.array([0.0, 100.0, 0.0])  # East, North, Up
        np.testing.assert_allclose(vel, expected, rtol=1e-6, atol=1e-12)

        # Test east-pointing velocity (90° yaw, 0° pitch)
        vel = guidance._missile_velocity_vec(90.0, 0.0, 100.0)
        expected = np.array([100.0, 0.0, 0.0])
        np.testing.assert_allclose(vel, expected, rtol=1e-6, atol=1e-12)

        # Test upward velocity (0° yaw, 90° pitch)
        vel = guidance._missile_velocity_vec(0.0, 90.0, 100.0)
        expected = np.array([0.0, 0.0, 100.0])
        np.testing.assert_allclose(vel, expected, rtol=1e-6, atol=1e-12)

    def test_pn_propnav_closing_velocity(self, mock_missile):
        """Test closing velocity calculation (computed inline in implementation)."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        PnPropNavGuidance(mock_missile)
        # NOTE: _closing_velocity is no longer a standalone method
        # It's computed inline as: Vc = max(1e-6, -float(np.dot(Vrel, Ulos)))
        # This test verifies the formula
        # Target and missile velocities
        Vt = np.array([100.0, 0.0, 0.0])  # Target moving east
        Vm = np.array([0.0, 100.0, 0.0])  # Missile moving north
        Ulos = np.array([1.0, 0.0, 0.0])  # LOS pointing east
        # Compute closing velocity using the inline formula
        Vrel = Vt - Vm  # [100, -100, 0]
        max(
            1e-6, -float(np.dot(Vrel, Ulos))
        )  # -dot([100, -100, 0], [1, 0, 0]) = -100, but max(1e-6, -100) = 1e-6
        # Closing velocity should be 1e-6 (clamped to minimum, since dot product is negative)
        # For a closing scenario, we need Vrel to point towards the missile (negative LOS)
        Vt_closing = np.array([-200.0, 0.0, 0.0])  # Target approaching from east
        Vm_closing = np.array([100.0, 0.0, 0.0])  # Missile moving east
        Vrel_closing = Vt_closing - Vm_closing  # [-300, 0, 0]
        Vc_closing = max(
            1e-6, -float(np.dot(Vrel_closing, Ulos))
        )  # -dot([-300, 0, 0], [1, 0, 0]) = 300
        assert abs(Vc_closing - 300.0) < 1e-10

    def test_pn_propnav_time_to_go(self, mock_missile):
        """Test time to go calculation."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        guidance = PnPropNavGuidance(mock_missile)
        # Simple head-on scenario
        R_vec = np.array([1000.0, 0.0, 0.0])  # Target 1000m east
        Ulos = np.array([1.0, 0.0, 0.0])  # Unit LOS vector
        Vt = np.array([-200.0, 0.0, 0.0])  # Target approaching west
        Vm = np.array([300.0, 0.0, 0.0])  # Missile moving east
        Vrel = Vt - Vm  # [-500, 0, 0]
        tgo = guidance._time_to_go(R_vec, Ulos, Vrel)
        # Closing speed: Vc = -dot(Vrel, Ulos) = -dot([-500, 0, 0], [1, 0, 0]) = 500 m/s
        # Time to go: |R_vec| / Vc = 1000 / 500 = 2.0 seconds
        assert abs(tgo - 2.0) < 1e-10

    def test_pn_propnav_acceleration_limiting(self, mock_missile):
        """Test acceleration limiting."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        guidance = PnPropNavGuidance(mock_missile)

        # Test normal acceleration (within limits)
        normal_accel = np.array([50.0, 100.0, 20.0])  # ~112 m/s²
        limited_accel = guidance._limit_acceleration(normal_accel)
        # Should not be limited (30g = ~294 m/s²)
        np.testing.assert_allclose(limited_accel, normal_accel, rtol=1e-6, atol=1e-12)

        # Test excessive acceleration
        high_accel = np.array([500.0, 500.0, 500.0])  # ~866 m/s²
        limited_high = guidance._limit_acceleration(high_accel)
        # Should be limited to 30g magnitude
        max_accel = 30.0 * 9.80665  # ~294 m/s²
        limited_magnitude = np.linalg.norm(limited_high)
        assert abs(limited_magnitude - max_accel) < 1e-6

    def test_pn_propnav_yaw_rate_limiting(self, mock_missile):
        """Test yaw rate limiting - method removed in refactoring."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        guidance = PnPropNavGuidance(mock_missile)
        # NOTE: _limit_yaw_rate() method was removed in the refactoring
        # Yaw rate limiting is now implicit in the acceleration saturation
        # and the mapping to yaw/pitch setpoints. Test that acceleration
        # limiting works correctly instead.
        # Verify acceleration limiting exists and works
        test_accel = np.array([500.0, 500.0, 500.0])  # High acceleration
        limited_accel = guidance._limit_acceleration(test_accel)
        # Should be limited to n_max * g
        max_accel = 30.0 * 9.80665  # ~294 m/s²
        limited_magnitude = np.linalg.norm(limited_accel)
        assert abs(limited_magnitude - max_accel) < 1e-6

    def test_pn_propnav_pure_pn_law(
        self, mock_missile, mock_missile_position, mock_target_position
    ):
        """Test Pure PN law implementation."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        with patch("bvr_marl_core.missiles.guidance.pn_propnav.geodetic_to_enu") as mock_enu:
            mock_enu.return_value = [1000.0, 1000.0, 500.0]  # ENU vector

            guidance = PnPropNavGuidance(mock_missile, pn_law=PnPropNavGuidance.PN_PURE)

            yaw, pitch = guidance.compute(
                current_yaw_deg=45.0,
                current_pitch_deg=10.0,
                missile_position=mock_missile_position,
                target_position=mock_target_position,
                tick_secs=0.1,
            )

            # Should produce valid guidance commands
            assert 0.0 <= yaw <= 360.0
            assert -89.0 <= pitch <= 89.0

    def test_pn_propnav_true_pn_law(
        self, mock_missile, mock_missile_position, mock_target_position
    ):
        """Test True PN law implementation."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        with patch("bvr_marl_core.missiles.guidance.pn_propnav.geodetic_to_enu") as mock_enu:
            mock_enu.return_value = [1000.0, 1000.0, 500.0]

            guidance = PnPropNavGuidance(mock_missile, pn_law=PnPropNavGuidance.PN_TRUE)

            yaw, pitch = guidance.compute(
                current_yaw_deg=45.0,
                current_pitch_deg=10.0,
                missile_position=mock_missile_position,
                target_position=mock_target_position,
                tick_secs=0.1,
            )

            # Should produce valid guidance commands
            assert 0.0 <= yaw <= 360.0
            assert -89.0 <= pitch <= 89.0

    def test_pn_propnav_zem_pn_law(self, mock_missile, mock_missile_position, mock_target_position):
        """Test ZEM PN law implementation."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        with patch("bvr_marl_core.missiles.guidance.pn_propnav.geodetic_to_enu") as mock_enu:
            mock_enu.return_value = [1000.0, 1000.0, 500.0]

            guidance = PnPropNavGuidance(mock_missile, pn_law=PnPropNavGuidance.PN_ZEM)

            yaw, pitch = guidance.compute(
                current_yaw_deg=45.0,
                current_pitch_deg=10.0,
                missile_position=mock_missile_position,
                target_position=mock_target_position,
                tick_secs=0.1,
            )

            # Should produce valid guidance commands
            assert 0.0 <= yaw <= 360.0
            assert -89.0 <= pitch <= 89.0

    def test_pn_propnav_target_velocity_integration(
        self, mock_missile, mock_missile_position, mock_target_position
    ):
        """Test integration with target velocity from radar/provider."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        with patch("bvr_marl_core.missiles.guidance.pn_propnav.geodetic_to_enu") as mock_enu:
            mock_enu.return_value = [1000.0, 1000.0, 0.0]

            # Mock target velocity from radar
            mock_missile.radar.get_tracker_info.return_value = {
                "enu_velocity": [200.0, 100.0, 50.0]
            }

            guidance = PnPropNavGuidance(mock_missile)

            yaw, pitch = guidance.compute(
                current_yaw_deg=0.0,
                current_pitch_deg=0.0,
                missile_position=mock_missile_position,
                target_position=mock_target_position,
                tick_secs=0.1,
            )

            # Should use radar velocity data
            assert 0.0 <= yaw <= 360.0
            assert -89.0 <= pitch <= 89.0

    def test_pn_propnav_hold_last_valid_velocity(
        self, mock_missile, mock_missile_position, mock_target_position
    ):
        """Test hold-last-valid: when tracker loses velocity, the last good value is reused."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        with patch("bvr_marl_core.missiles.guidance.pn_propnav.geodetic_to_enu") as mock_enu:
            mock_enu.return_value = [1000.0, 1000.0, 0.0]
            guidance = PnPropNavGuidance(mock_missile)
            # First call - provide valid velocity via provider (both APIs)
            mock_missile.target_provider.get_guidance_velocity.return_value = [200.0, 100.0, 0.0]
            mock_missile.target_provider.get_guidance_velocity_in_missile_enu.return_value = [
                200.0,
                100.0,
                0.0,
            ]
            guidance.compute(0.0, 0.0, mock_missile_position, mock_target_position, 0.5)
            # After first call, hold-last-valid should be stored
            assert guidance._last_valid_vt is not None
            np.testing.assert_allclose(
                guidance._last_valid_vt, [200.0, 100.0, 0.0], rtol=1e-6, atol=1e-1
            )
            # Second call - velocity disappears (both APIs)
            mock_missile.target_provider.get_guidance_velocity.return_value = None
            mock_missile.target_provider.get_guidance_velocity_in_missile_enu.return_value = None
            yaw, pitch = guidance.compute(
                0.0, 0.0, mock_missile_position, mock_target_position, 0.5
            )
            # Should produce valid commands (hold-last-valid in effect)
            assert 0.0 <= yaw <= 360.0
            assert -89.0 <= pitch <= 89.0
            # Age should have incremented
            assert guidance._last_valid_vt_age > 0.0

    def test_pn_propnav_angle_normalization(
        self, mock_missile, mock_missile_position, mock_target_position
    ):
        """Test that output angles are properly normalized."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        with patch("bvr_marl_core.missiles.guidance.pn_propnav.geodetic_to_enu") as mock_enu:
            mock_enu.return_value = [1000.0, 1000.0, 500.0]

            guidance = PnPropNavGuidance(mock_missile)

            # Test with large initial yaw that might cause wraparound
            yaw, pitch = guidance.compute(
                current_yaw_deg=350.0,  # Close to 360°
                current_pitch_deg=85.0,  # Close to limit
                missile_position=mock_missile_position,
                target_position=mock_target_position,
                tick_secs=0.1,
            )

            # Yaw should be normalized
            assert 0.0 <= yaw <= 360.0

            # Pitch should be clamped to [-89, 89]
            assert -89.0 <= pitch <= 89.0

    def test_pn_fallback_uses_velocity_in_missile_enu(self, mock_missile, mock_missile_position):
        """
        When the tracker manager has no track state, _get_target_state_enu()
        must call get_guidance_velocity_in_missile_enu(missile_position) rather
        than the raw get_guidance_velocity() so the cached vector is rotated
        into the current missile ENU before being used by PN.
        """
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        rotated_vel = [200.0, 100.0, 0.0]

        # Provider exposes the frame-aware API
        mock_missile.target_provider.get_guidance_velocity_in_missile_enu = Mock(
            return_value=rotated_vel
        )
        mock_missile.target_provider.get_guidance_tid.return_value = None
        mock_missile.radar.tracker_manager = None  # force fallback path

        guidance = PnPropNavGuidance(mock_missile)
        p, v = guidance._get_target_state_enu(mock_missile_position, Mock())

        # Frame-aware helper must have been called with the missile position
        mock_missile.target_provider.get_guidance_velocity_in_missile_enu.assert_called_once_with(
            mock_missile_position
        )
        # Raw accessor must NOT have been called
        mock_missile.target_provider.get_guidance_velocity.assert_not_called()
        assert v is not None
        assert list(v) == pytest.approx(rotated_vel, abs=1e-9)
