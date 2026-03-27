from unittest.mock import Mock, patch

import pytest


class TestTerminalGuidance:
    """Test the terminal guidance system."""

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

    def test_terminal_guidance_import(self):
        """Test that terminal guidance can be imported."""
        from air_to_air_rl.missiles.guidance.terminal import TerminalGuidance

        assert TerminalGuidance is not None

    def test_terminal_guidance_initialization(self, mock_missile):
        """Test terminal guidance initialization."""
        from air_to_air_rl.missiles.guidance.terminal import TerminalGuidance

        with (
            patch("air_to_air_rl.missiles.guidance.terminal.DirectPursuitGuidance") as mock_direct,
            patch("air_to_air_rl.missiles.guidance.terminal.PnPropNavGuidance") as mock_pn,
        ):
            guidance = TerminalGuidance(mock_missile)

            assert guidance.missile == mock_missile
            assert hasattr(guidance, "direct_guidance")
            assert hasattr(guidance, "pn")

            # Verify that component guidance systems were initialized
            mock_direct.assert_called_once_with(mock_missile)
            mock_pn.assert_called_once_with(mock_missile)

    def test_terminal_guidance_compute_uses_direct(
        self, mock_missile, mock_missile_position, mock_target_position
    ):
        """Test that terminal guidance uses direct pursuit guidance."""
        from air_to_air_rl.missiles.guidance.terminal import TerminalGuidance

        with (
            patch(
                "air_to_air_rl.missiles.guidance.terminal.DirectPursuitGuidance"
            ) as mock_direct_class,
            patch("air_to_air_rl.missiles.guidance.terminal.PnPropNavGuidance") as mock_pn_class,
        ):
            # Set up mock direct guidance
            mock_direct_instance = Mock()
            mock_direct_instance.compute.return_value = (90.0, 15.0)
            mock_direct_class.return_value = mock_direct_instance

            # Set up mock PN guidance (shouldn't be called)
            mock_pn_instance = Mock()
            mock_pn_class.return_value = mock_pn_instance

            guidance = TerminalGuidance(mock_missile)

            yaw, pitch = guidance.compute(
                current_yaw_deg=45.0,
                current_pitch_deg=10.0,
                missile_position=mock_missile_position,
                target_position=mock_target_position,
                tick_secs=0.1,
            )

            # Should return direct guidance result
            assert yaw == 90.0
            assert pitch == 15.0

            # Verify direct guidance was called
            mock_direct_instance.compute.assert_called_once_with(
                45.0, 10.0, mock_missile_position, mock_target_position, 0.1
            )

            # Verify PN guidance was NOT called
            mock_pn_instance.compute.assert_not_called()

    def test_terminal_guidance_passthrough_parameters(
        self, mock_missile, mock_missile_position, mock_target_position
    ):
        """Test that terminal guidance passes through all parameters correctly."""
        from air_to_air_rl.missiles.guidance.terminal import TerminalGuidance

        with patch(
            "air_to_air_rl.missiles.guidance.terminal.DirectPursuitGuidance"
        ) as mock_direct_class:
            mock_direct_instance = Mock()
            mock_direct_instance.compute.return_value = (120.0, -5.0)
            mock_direct_class.return_value = mock_direct_instance

            guidance = TerminalGuidance(mock_missile)

            # Test with different parameter values
            current_yaw = 270.0
            current_pitch = 20.0
            tick_secs = 0.05

            yaw, pitch = guidance.compute(
                current_yaw_deg=current_yaw,
                current_pitch_deg=current_pitch,
                missile_position=mock_missile_position,
                target_position=mock_target_position,
                tick_secs=tick_secs,
            )

            # Verify all parameters were passed correctly
            mock_direct_instance.compute.assert_called_once_with(
                current_yaw, current_pitch, mock_missile_position, mock_target_position, tick_secs
            )

            assert yaw == 120.0
            assert pitch == -5.0

    def test_terminal_guidance_multiple_calls(
        self, mock_missile, mock_missile_position, mock_target_position
    ):
        """Test that terminal guidance works correctly with multiple calls."""
        from air_to_air_rl.missiles.guidance.terminal import TerminalGuidance

        with patch(
            "air_to_air_rl.missiles.guidance.terminal.DirectPursuitGuidance"
        ) as mock_direct_class:
            mock_direct_instance = Mock()
            # Different return values for multiple calls
            mock_direct_instance.compute.side_effect = [
                (45.0, 5.0),  # First call
                (50.0, 7.0),  # Second call
                (55.0, 9.0),  # Third call
            ]
            mock_direct_class.return_value = mock_direct_instance

            guidance = TerminalGuidance(mock_missile)

            # First call
            yaw1, pitch1 = guidance.compute(
                0.0, 0.0, mock_missile_position, mock_target_position, 0.1
            )
            assert yaw1 == 45.0
            assert pitch1 == 5.0

            # Second call
            yaw2, pitch2 = guidance.compute(
                10.0, 2.0, mock_missile_position, mock_target_position, 0.1
            )
            assert yaw2 == 50.0
            assert pitch2 == 7.0

            # Third call
            yaw3, pitch3 = guidance.compute(
                20.0, 4.0, mock_missile_position, mock_target_position, 0.1
            )
            assert yaw3 == 55.0
            assert pitch3 == 9.0

            # Verify all calls were made
            assert mock_direct_instance.compute.call_count == 3

    def test_terminal_guidance_component_integration(self, mock_missile):
        """Test that terminal guidance properly integrates component guidance systems."""
        from air_to_air_rl.missiles.guidance.terminal import TerminalGuidance

        guidance = TerminalGuidance(mock_missile)

        # Verify that both guidance components are accessible
        assert hasattr(guidance, "direct_guidance")
        assert hasattr(guidance, "pn")
        assert guidance.direct_guidance is not None
        assert guidance.pn is not None

    def test_terminal_guidance_future_pn_integration(
        self, mock_missile, mock_missile_position, mock_target_position
    ):
        """Test framework for future PN guidance integration."""
        from air_to_air_rl.missiles.guidance.terminal import TerminalGuidance

        with (
            patch("air_to_air_rl.missiles.guidance.terminal.DirectPursuitGuidance"),
            patch("air_to_air_rl.missiles.guidance.terminal.PnPropNavGuidance") as mock_pn_class,
        ):
            guidance = TerminalGuidance(mock_missile)

            # Verify PN guidance is initialized (for future use)
            assert guidance.pn is not None
            mock_pn_class.assert_called_once_with(mock_missile)

            # Current implementation should still use direct guidance
            # but PN is available for future enhancement
