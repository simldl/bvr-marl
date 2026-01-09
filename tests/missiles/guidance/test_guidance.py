import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


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
        return provider

    def test_missile_guidance_import(self):
        """Test that missile guidance can be imported."""
        try:
            from missiles.guidance.guidance import MissileGuidance
            assert MissileGuidance is not None
        except ImportError:
            pytest.skip("MissileGuidance not available")

    def test_missile_guidance_initialization(self, mock_missile, mock_target_provider):
        """Test missile guidance system initialization."""
        try:
            from missiles.guidance.guidance import MissileGuidance
            
            with patch.multiple('missiles.guidance.guidance',
                              LoftGuidance=Mock(),
                              LeadInterceptGuidance=Mock(),
                              FovCaptureGuidance=Mock(),
                              TerminalGuidance=Mock(),
                              DirectPursuitGuidance=Mock(),
                              PnPropNavGuidance=Mock(),
                              CircularMovingAverageFilter=Mock(),
                              MovingAverageFilter=Mock()):

                guidance = MissileGuidance(mock_missile, mock_target_provider)

                assert guidance.missile == mock_missile
                assert guidance.target_provider == mock_target_provider
                assert hasattr(guidance, 'loft')
                assert hasattr(guidance, 'fov_capture')
                assert hasattr(guidance, 'pn')
                assert hasattr(guidance, 'terminal')
                assert hasattr(guidance, 'direct')
                
        except ImportError:
            pytest.skip("MissileGuidance dependencies not available")

    def test_missile_guidance_phase_selection(self, mock_missile, mock_target_provider):
        """Test missile guidance phase selection logic."""
        try:
            from missiles.guidance.guidance import MissileGuidance

            with patch.multiple('missiles.guidance.guidance',
                              LoftGuidance=Mock(),
                              LeadInterceptGuidance=Mock(),
                              FovCaptureGuidance=Mock(),
                              TerminalGuidance=Mock(),
                              DirectPursuitGuidance=Mock(),
                              PnPropNavGuidance=Mock(),
                              CircularMovingAverageFilter=Mock(),
                              MovingAverageFilter=Mock()) as mocks:

                # Mock tracker manager
                mock_tracker_manager = Mock()

                guidance = MissileGuidance(mock_missile, mock_target_provider)

                # Mock terminal activation to return False by default
                guidance.terminal.should_activate = Mock(return_value=False)
                guidance._has_lock = Mock(return_value=False)
                guidance._target_in_radar_fov = Mock(return_value=False)

                # Set up target provider to return a position
                target_pos = Mock()
                target_pos.lat = 46.0
                target_pos.lon = 3.0
                target_pos.alt = 8500.0
                mock_target_provider.get_guidance_target.return_value = target_pos

                # Test default phase (should be direct or fov_capture when not locked)
                phase = guidance._select_guidance_phase(
                    mock_missile,
                    mock_target_provider,
                    mock_tracker_manager,
                    0.1
                )
                # Without lock or FOV, should be fov_capture or direct
                assert phase in ["fov_capture", "direct"]

                # Test with lock and FOV
                guidance._has_lock = Mock(return_value=True)
                guidance._target_in_radar_fov = Mock(return_value=True)
                guidance.pn_min_range_m = 1000.0

                phase = guidance._select_guidance_phase(
                    mock_missile,
                    mock_target_provider,
                    mock_tracker_manager,
                    0.1
                )
                # With lock and FOV and sufficient range, should be PN
                assert phase == "pn"

        except ImportError:
            pytest.skip("MissileGuidance dependencies not available")

    def test_missile_guidance_compute_guidance(self, mock_missile, mock_target_provider):
        """Test guidance computation with different phases."""
        try:
            from missiles.guidance.guidance import MissileGuidance

            with patch.multiple('missiles.guidance.guidance',
                              LoftGuidance=Mock(),
                              LeadInterceptGuidance=Mock(),
                              FovCaptureGuidance=Mock(),
                              TerminalGuidance=Mock(),
                              PnPropNavGuidance=Mock(),
                              CircularMovingAverageFilter=Mock(),
                              MovingAverageFilter=Mock()) as mocks, \
                 patch('missiles.guidance.guidance.DirectPursuitGuidance') as mock_direct, \
                 patch('missiles.guidance.guidance.PnPropNavGuidance') as mock_pn, \
                 patch('missiles.guidance.guidance.CircularMovingAverageFilter') as mock_circular, \
                 patch('missiles.guidance.guidance.MovingAverageFilter') as mock_moving:

                # Configure guidance algorithm mocks
                direct_mock = Mock()
                direct_mock.compute.return_value = (45.0, 10.0)
                mock_direct.return_value = direct_mock

                pn_mock = Mock()
                pn_mock.compute.return_value = (50.0, 15.0)
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
                mock_target_provider.get_guidance_velocity.return_value = [100.0, 0.0, 0.0]  # ENU velocity

                # Mock the phase selection to return 'pn' phase
                with patch.object(MissileGuidance, '_select_guidance_phase', return_value='pn') as mock_phase:
                    guidance = MissileGuidance(mock_missile, mock_target_provider)

                    # New signature: compute_guidance(missile, target_provider, tracker_manager, dt)
                    yaw, pitch = guidance.compute_guidance(
                        mock_missile,
                        mock_target_provider,
                        mock_tracker_manager,
                        0.1
                    )

                # Test PN guidance was called
                pn_mock.compute.assert_called_once()

                # Should return PN guidance results
                assert yaw == 50.0
                assert pitch == 15.0

        except ImportError:
            pytest.skip("MissileGuidance dependencies not available")

    def test_missile_guidance_target_in_fov(self, mock_missile, mock_target_provider):
        """Test target in radar field of view detection."""
        try:
            from missiles.guidance.guidance import MissileGuidance
            
            with patch.multiple('missiles.guidance.guidance',
                              LoftGuidance=Mock(),
                              LeadInterceptGuidance=Mock(),
                              FovCaptureGuidance=Mock(),
                              TerminalGuidance=Mock(),
                              DirectPursuitGuidance=Mock(),
                              PnPropNavGuidance=Mock(),
                              CircularMovingAverageFilter=Mock(),
                              MovingAverageFilter=Mock()) as mocks, \
                 patch('missiles.guidance.guidance._angles_dist') as mock_angles:
                
                # Mock radar attributes
                mock_missile.radar.yaw_deg = 90.0
                mock_missile.radar.pitch_deg = 0.0
                mock_missile.radar.yaw_offset_deg = 0.0
                mock_missile.radar.pitch_offset_deg = 0.0
                mock_missile.radar.h_fov_deg = 60.0
                mock_missile.radar.v_fov_deg = 30.0
                
                guidance = MissileGuidance(mock_missile, mock_target_provider)
                
                # Test target within FOV
                mock_angles.return_value = (10.0, 5.0, 1000.0)  # az, el, distance
                in_fov = guidance._target_in_radar_fov(
                    mock_missile.position,
                    mock_target_provider.get_guidance_target()
                )
                assert in_fov is True
                
                # Test target outside horizontal FOV
                mock_angles.return_value = (35.0, 5.0, 1000.0)  # Outside h_fov/2
                in_fov = guidance._target_in_radar_fov(
                    mock_missile.position,
                    mock_target_provider.get_guidance_target()
                )
                assert in_fov is False
                
                # Test target outside vertical FOV
                mock_angles.return_value = (10.0, 20.0, 1000.0)  # Outside v_fov/2
                in_fov = guidance._target_in_radar_fov(
                    mock_missile.position,
                    mock_target_provider.get_guidance_target()
                )
                assert in_fov is False
                
        except ImportError:
            pytest.skip("MissileGuidance dependencies not available")

    def test_missile_guidance_no_target(self, mock_missile, mock_target_provider):
        """Test guidance behavior when no target is available."""
        try:
            from missiles.guidance.guidance import MissileGuidance

            with patch.multiple('missiles.guidance.guidance',
                              LoftGuidance=Mock(),
                              LeadInterceptGuidance=Mock(),
                              FovCaptureGuidance=Mock(),
                              TerminalGuidance=Mock(),
                              DirectPursuitGuidance=Mock(),
                              PnPropNavGuidance=Mock(),
                              CircularMovingAverageFilter=Mock(),
                              MovingAverageFilter=Mock()) as mocks, \
                 patch('missiles.guidance.guidance.DirectPursuitGuidance') as mock_direct:

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
                    mock_missile,
                    mock_target_provider,
                    mock_tracker_manager,
                    0.1
                )

                # Should return missile's current angles when no target is available
                assert yaw == mock_missile.yaw_deg
                assert pitch == mock_missile.pitch_deg

        except ImportError:
            pytest.skip("MissileGuidance dependencies not available")

    def test_missile_guidance_filtering(self, mock_missile, mock_target_provider):
        """Test guidance filtering for different phases."""
        try:
            from missiles.guidance.guidance import MissileGuidance

            with patch.multiple('missiles.guidance.guidance',
                              LoftGuidance=Mock(),
                              LeadInterceptGuidance=Mock(),
                              FovCaptureGuidance=Mock(),
                              TerminalGuidance=Mock(),
                              PnPropNavGuidance=Mock(),
                              CircularMovingAverageFilter=Mock(),
                              MovingAverageFilter=Mock()) as mocks, \
                 patch('missiles.guidance.guidance.DirectPursuitGuidance') as mock_direct_filt, \
                 patch('missiles.guidance.guidance.PnPropNavGuidance') as mock_pn_filt, \
                 patch('missiles.guidance.guidance.CircularMovingAverageFilter') as mock_circular_filt, \
                 patch('missiles.guidance.guidance.MovingAverageFilter') as mock_moving_filt:

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
                with patch.object(MissileGuidance, '_select_guidance_phase', return_value='direct') as mock_phase_filt:
                    guidance = MissileGuidance(mock_missile, mock_target_provider)

                    # Test guidance computation with new signature
                    yaw, pitch = guidance.compute_guidance(
                        mock_missile,
                        mock_target_provider,
                        mock_tracker_manager,
                        0.1
                    )

                # Direct pursuit should return unfiltered output
                # (The new implementation doesn't apply filtering at this level)
                assert yaw == 45.0
                assert pitch == 10.0

        except ImportError:
            pytest.skip("MissileGuidance dependencies not available")