from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest


class TestMissileGuidance:
    """Test the main missile guidance system."""

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
        missile.radar = Mock()
        missile.radar.get_locked_target.return_value = None
        missile.radar.get_tracks.return_value = []
        missile.radar.h_fov_deg = 60.0
        missile.radar.v_fov_deg = 30.0
        missile.phase_manager = Mock()
        missile.phase_manager.current_phase = "active"
        return missile

    @pytest.fixture
    def mock_target_provider(self):
        """Create a mock target provider."""
        provider = Mock()
        provider.get_guidance_target.return_value = Mock()
        provider.get_guidance_target.return_value.lat = 46.0
        provider.get_guidance_target.return_value.lon = 3.0
        provider.get_guidance_target.return_value.alt = 8500.0
        provider.get_guidance_tid.return_value = None  # no track by default
        provider.has_fresh_track.return_value = False
        provider.has_coastable_track.return_value = False
        return provider

    def test_missile_guidance_import(self):
        """Test that missile guidance can be imported."""
        from bvr_marl_core.missiles.guidance.guidance import MissileGuidance

        assert MissileGuidance is not None

    def test_missile_guidance_initialization(self, mock_missile, mock_target_provider):
        """Test missile guidance system initialization."""
        from bvr_marl_core.missiles.guidance.guidance import MissileGuidance

        with patch.multiple(
            "bvr_marl_core.missiles.guidance.guidance",
            LoftGuidance=Mock(),
            LeadInterceptGuidance=Mock(),
            DirectPursuitGuidance=Mock(),
            PnPropNavGuidance=Mock(),
            CircularMovingAverageFilter=Mock(),
            MovingAverageFilter=Mock(),
        ):
            guidance = MissileGuidance(mock_missile, mock_target_provider)
            assert guidance.missile == mock_missile
            assert guidance.target_provider == mock_target_provider
            assert hasattr(guidance, "loft")
            assert hasattr(guidance, "pn")
            assert hasattr(guidance, "direct")

    def test_missile_guidance_phase_selection(self, mock_missile, mock_target_provider):
        """Test missile guidance phase selection logic."""
        from bvr_marl_core.missiles.guidance.guidance import MissileGuidance

        with patch.multiple(
            "bvr_marl_core.missiles.guidance.guidance",
            LoftGuidance=Mock(),
            LeadInterceptGuidance=Mock(),
            DirectPursuitGuidance=Mock(),
            PnPropNavGuidance=Mock(),
            CircularMovingAverageFilter=Mock(),
            MovingAverageFilter=Mock(),
        ):
            # Mock tracker manager
            mock_tracker_manager = Mock()
            guidance = MissileGuidance(mock_missile, mock_target_provider)
            guidance._has_lock = Mock(return_value=False)
            # Set up target provider to return a position
            target_pos = Mock()
            target_pos.lat = 46.0
            target_pos.lon = 3.0
            target_pos.alt = 8500.0
            mock_target_provider.get_guidance_target.return_value = target_pos
            # Test default phase: no velocity track falls back to direct.
            phase = guidance._select_guidance_phase(
                mock_missile, mock_target_provider, mock_tracker_manager, 0.1
            )
            assert phase == "direct"
            # Test with lock but no fresh track
            guidance._has_lock = Mock(return_value=True)
            phase = guidance._select_guidance_phase(
                mock_missile, mock_target_provider, mock_tracker_manager, 0.1
            )
            # Lock alone is not enough for velocity-based PN.
            assert phase == "direct"

            mock_target_provider.has_fresh_track.return_value = True
            phase = guidance._select_guidance_phase(
                mock_missile, mock_target_provider, mock_tracker_manager, 0.1
            )
            assert phase == "pn"

            guidance._has_lock = Mock(return_value=False)
            phase = guidance._select_guidance_phase(
                mock_missile, mock_target_provider, mock_tracker_manager, 0.1
            )
            assert phase == "lead"

    def test_terminal_guidance_uses_short_same_target_coast(
        self, mock_missile, mock_target_provider
    ):
        """Brief terminal track outages should keep velocity-aware guidance."""
        from bvr_marl_core.missiles.guidance.guidance import MissileGuidance

        with patch.multiple(
            "bvr_marl_core.missiles.guidance.guidance",
            LoftGuidance=Mock(),
            LeadInterceptGuidance=Mock(),
            DirectPursuitGuidance=Mock(),
            PnPropNavGuidance=Mock(),
            CircularMovingAverageFilter=Mock(),
            MovingAverageFilter=Mock(),
        ):
            guidance = MissileGuidance(mock_missile, mock_target_provider)
            guidance._has_lock = Mock(return_value=True)

            close_target = Mock()
            close_target.lat = mock_missile.position.lat
            close_target.lon = mock_missile.position.lon + 0.005
            close_target.alt = mock_missile.position.alt
            mock_target_provider.get_guidance_target.return_value = close_target
            mock_target_provider.has_fresh_track.return_value = False
            mock_target_provider.has_coastable_track.return_value = True

            phase = guidance._select_guidance_phase(mock_missile, mock_target_provider, Mock(), 0.1)

            assert phase == "terminal"

            far_target = Mock()
            far_target.lat = mock_missile.position.lat + 1.0
            far_target.lon = mock_missile.position.lon + 1.0
            far_target.alt = mock_missile.position.alt
            mock_target_provider.get_guidance_target.return_value = far_target

            phase = guidance._select_guidance_phase(mock_missile, mock_target_provider, Mock(), 0.1)

            assert phase == "direct"

    def test_missile_guidance_compute_guidance(self, mock_missile, mock_target_provider):
        """Test guidance computation with different phases."""
        from bvr_marl_core.missiles.guidance.guidance import MissileGuidance

        with (
            patch.multiple(
                "bvr_marl_core.missiles.guidance.guidance",
                LoftGuidance=Mock(),
                LeadInterceptGuidance=Mock(),
                PnPropNavGuidance=Mock(),
                CircularMovingAverageFilter=Mock(),
                MovingAverageFilter=Mock(),
            ),
            patch("bvr_marl_core.missiles.guidance.guidance.DirectPursuitGuidance") as mock_direct,
            patch("bvr_marl_core.missiles.guidance.guidance.PnPropNavGuidance") as mock_pn,
            patch(
                "bvr_marl_core.missiles.guidance.guidance.CircularMovingAverageFilter"
            ) as mock_circular,
            patch("bvr_marl_core.missiles.guidance.guidance.MovingAverageFilter") as mock_moving,
        ):
            # Configure guidance algorithm mocks
            direct_mock = Mock()
            direct_mock.compute.return_value = (45.0, 10.0)
            mock_direct.return_value = direct_mock
            pn_mock = Mock()
            pn_mock.compute.return_value = (50.0, 15.0)
            pn_mock.terminal_range_m = 2000.0
            mock_pn.return_value = pn_mock
            # Configure filters
            mock_yaw_filter = Mock()
            mock_yaw_filter.average.return_value = 47.5
            mock_pitch_filter = Mock()
            mock_pitch_filter.average.return_value = 12.5
            mock_circular.return_value = mock_yaw_filter
            mock_moving.return_value = mock_pitch_filter
            # Mock tracker manager
            mock_tracker_manager = Mock()
            # Mock target position and velocity for PN guidance
            target_pos = Mock()
            target_pos.lat = 46.0
            target_pos.lon = 3.0
            target_pos.alt = 8500.0
            mock_target_provider.get_guidance_target.return_value = target_pos
            mock_target_provider.get_guidance_velocity.return_value = [
                100.0,
                0.0,
                0.0,
            ]  # ENU velocity
            # Mock the phase selection to return 'pn' phase
            with patch.object(MissileGuidance, "_select_guidance_phase", return_value="pn"):
                guidance = MissileGuidance(mock_missile, mock_target_provider)
                # New signature: compute_guidance(missile, target_provider, tracker_manager, dt)
                yaw, pitch = guidance.compute_guidance(
                    mock_missile, mock_target_provider, mock_tracker_manager, 0.1
                )
            # Test PN guidance was called
            pn_mock.compute.assert_called_once()
            # Should return PN guidance results
            assert yaw == 50.0
            assert pitch == 15.0

    def test_missile_guidance_no_target(self, mock_missile, mock_target_provider):
        """Test guidance behavior when no target is available."""
        from bvr_marl_core.missiles.guidance.guidance import MissileGuidance

        with (
            patch.multiple(
                "bvr_marl_core.missiles.guidance.guidance",
                LoftGuidance=Mock(),
                LeadInterceptGuidance=Mock(),
                DirectPursuitGuidance=Mock(),
                PnPropNavGuidance=Mock(),
                CircularMovingAverageFilter=Mock(),
                MovingAverageFilter=Mock(),
            ),
            patch("bvr_marl_core.missiles.guidance.guidance.DirectPursuitGuidance") as mock_direct,
        ):
            # Configure direct pursuit to return fallback values
            direct_mock = Mock()
            direct_mock.compute.return_value = (45.0, 10.0)
            mock_direct.return_value = direct_mock
            # Configure target provider to return None for target position
            mock_target_provider.get_guidance_target.side_effect = Exception("No target")
            mock_tracker_manager = Mock()
            guidance = MissileGuidance(mock_missile, mock_target_provider)
            # Should fall back to current angles when no target
            yaw, pitch = guidance.compute_guidance(
                mock_missile, mock_target_provider, mock_tracker_manager, 0.1
            )
            # Should return missile's current angles when no target is available
            assert yaw == mock_missile.yaw_deg
            assert pitch == mock_missile.pitch_deg

    def test_missile_guidance_filtering(self, mock_missile, mock_target_provider):
        """Test guidance filtering for different phases."""
        from bvr_marl_core.missiles.guidance.guidance import MissileGuidance

        with (
            patch.multiple(
                "bvr_marl_core.missiles.guidance.guidance",
                LoftGuidance=Mock(),
                LeadInterceptGuidance=Mock(),
                PnPropNavGuidance=Mock(),
                CircularMovingAverageFilter=Mock(),
                MovingAverageFilter=Mock(),
            ),
            patch(
                "bvr_marl_core.missiles.guidance.guidance.DirectPursuitGuidance"
            ) as mock_direct_filt,
            patch("bvr_marl_core.missiles.guidance.guidance.PnPropNavGuidance") as mock_pn_filt,
            patch(
                "bvr_marl_core.missiles.guidance.guidance.CircularMovingAverageFilter"
            ) as mock_circular_filt,
            patch(
                "bvr_marl_core.missiles.guidance.guidance.MovingAverageFilter"
            ) as mock_moving_filt,
        ):
            # Configure guidance algorithms
            direct_mock_filt = Mock()
            direct_mock_filt.compute.return_value = (45.0, 10.0)
            mock_direct_filt.return_value = direct_mock_filt
            pn_mock_filt = Mock()
            pn_mock_filt.compute.return_value = (50.0, 15.0)
            mock_pn_filt.return_value = pn_mock_filt
            # Configure filters
            mock_yaw_filter = Mock()
            mock_yaw_filter.average.return_value = 47.5
            mock_pitch_filter = Mock()
            mock_pitch_filter.average.return_value = 12.5
            mock_circular_filt.return_value = mock_yaw_filter
            mock_moving_filt.return_value = mock_pitch_filter
            mock_tracker_manager = Mock()
            # Mock target position
            target_pos = Mock()
            target_pos.lat = 46.0
            target_pos.lon = 3.0
            target_pos.alt = 8500.0
            mock_target_provider.get_guidance_target.return_value = target_pos
            # Mock the phase selection to return 'direct' phase
            with patch.object(MissileGuidance, "_select_guidance_phase", return_value="direct"):
                guidance = MissileGuidance(mock_missile, mock_target_provider)
                # Test guidance computation with new signature
                yaw, pitch = guidance.compute_guidance(
                    mock_missile, mock_target_provider, mock_tracker_manager, 0.1
                )
            # Direct pursuit should return unfiltered output
            # (The new implementation doesn't apply filtering at this level)
            assert yaw == 45.0
            assert pitch == 10.0

    def test_pn_state_resets_on_locked_target_change(self, mock_missile, mock_target_provider):
        """
        When radar.get_locked_target() returns a new unit ID, PN hold-last-valid
        state must be cleared immediately, even before the tracker TID changes.

        Keeping the cached velocity from target A while the seeker is locked on B
        produces a sideways guidance command toward the wrong intercept point.
        """
        from bvr_marl_core.missiles.guidance.guidance import MissileGuidance

        with patch.multiple(
            "bvr_marl_core.missiles.guidance.guidance",
            LoftGuidance=Mock(),
            LeadInterceptGuidance=Mock(),
            DirectPursuitGuidance=Mock(),
            CircularMovingAverageFilter=Mock(),
            MovingAverageFilter=Mock(),
        ):
            mock_missile.radar.get_locked_target = Mock(return_value="target-A")

            guidance = MissileGuidance(mock_missile, mock_target_provider)
            # Simulate live PN state for target A
            guidance.pn._last_valid_vt = np.array([200.0, 0.0, 0.0])
            guidance.pn._last_valid_vt_age = 0.5
            guidance._active_locked_id = "target-A"
            guidance.active_tid = None

            mock_tracker_manager = Mock()
            mock_target_provider.get_guidance_tid.return_value = None

            # Radar switches to target B
            mock_missile.radar.get_locked_target.return_value = "target-B"
            with patch.object(MissileGuidance, "_select_guidance_phase", return_value="direct"):
                guidance.compute_guidance(
                    mock_missile, mock_target_provider, mock_tracker_manager, 0.1
                )

            assert guidance._active_locked_id == "target-B"
            assert guidance.pn._last_valid_vt is None, (
                "PN cached velocity must be cleared when locked target changes."
            )
            assert guidance.pn._last_valid_vt_age == 0.0

    def test_missile_does_not_fly_toward_old_target_after_retarget(
        self, mock_missile, mock_target_provider
    ):
        """
        Integration: after guidance adopts the new locked target, the guidance
        target position returned by target_provider must reflect the new target,
        not the old one.

        This verifies that the sync in target_provider + PN reset in guidance
        work end-to-end: the missile's computed heading moves toward target B,
        not target A.
        """
        from bvr_marl_core.missiles.guidance.guidance import MissileGuidance

        with patch.multiple(
            "bvr_marl_core.missiles.guidance.guidance",
            LoftGuidance=Mock(),
            LeadInterceptGuidance=Mock(),
            CircularMovingAverageFilter=Mock(),
            MovingAverageFilter=Mock(),
        ):
            # Position for target B is North of missile (yaw ~0°)
            target_b_pos = Mock()
            target_b_pos.lat = 1.0  # North
            target_b_pos.lon = 0.0
            target_b_pos.alt = 8000.0

            # After retarget, provider returns B's position
            mock_target_provider.get_guidance_target.return_value = target_b_pos
            mock_target_provider.get_guidance_tid.return_value = None

            mock_missile.radar.get_locked_target = Mock(return_value="target-B")
            mock_missile.yaw_deg = 90.0  # currently pointing East
            mock_missile.pitch_deg = 0.0
            mock_missile.position = Mock(lat=0.0, lon=0.0, alt=8000.0)

            with patch(
                "bvr_marl_core.missiles.guidance.guidance.DirectPursuitGuidance"
            ) as MockDirect:
                direct_instance = Mock()
                direct_instance.compute.return_value = (5.0, 0.0)  # North-ish
                MockDirect.return_value = direct_instance

                guidance = MissileGuidance(mock_missile, mock_target_provider)
                guidance._active_locked_id = "target-B"

                mock_tracker_manager = Mock()
                with patch.object(MissileGuidance, "_select_guidance_phase", return_value="direct"):
                    yaw, _ = guidance.compute_guidance(
                        mock_missile, mock_target_provider, mock_tracker_manager, 0.1
                    )

            # Direct pursuit was called with the new target (B) position
            direct_instance.compute.assert_called_once()
            called_tgt = direct_instance.compute.call_args[0][3]  # 4th positional arg
            assert called_tgt is target_b_pos, (
                "Guidance must compute toward the new locked target B, not the old target A."
            )
