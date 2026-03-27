from unittest.mock import Mock

import pytest


class TestBaseGuidanceMode:
    """Test the base guidance mode class."""

    @pytest.fixture
    def mock_missile(self):
        """Create a mock missile."""
        missile = Mock()
        missile.position = Mock()
        missile.position.lat = 45.0
        missile.position.lon = 2.0
        missile.position.alt = 8000.0
        return missile

    def test_base_guidance_import(self):
        """Test that base guidance can be imported."""
        from air_to_air_rl.missiles.guidance.base import BaseGuidanceMode

        assert BaseGuidanceMode is not None

    def test_base_guidance_initialization(self, mock_missile):
        """Test base guidance initialization."""
        from air_to_air_rl.missiles.guidance.base import BaseGuidanceMode

        guidance = BaseGuidanceMode(mock_missile)
        assert guidance.missile == mock_missile

    def test_base_guidance_not_implemented(self, mock_missile):
        """Test that base guidance compute raises NotImplementedError."""
        from air_to_air_rl.missiles.guidance.base import BaseGuidanceMode

        guidance = BaseGuidanceMode(mock_missile)

        with pytest.raises(NotImplementedError):
            guidance.compute(
                current_yaw_deg=0.0,
                current_pitch_deg=0.0,
                missile_position=mock_missile.position,
                target_position=Mock(),
                tick_secs=0.1,
            )

    def test_base_guidance_inheritance_pattern(self, mock_missile):
        """Test that custom guidance modes can inherit from base."""
        from air_to_air_rl.missiles.guidance.base import BaseGuidanceMode

        class TestGuidanceMode(BaseGuidanceMode):
            def compute(
                self,
                current_yaw_deg,
                current_pitch_deg,
                missile_position,
                target_position,
                tick_secs,
            ):
                return current_yaw_deg + 10.0, current_pitch_deg + 5.0

        guidance = TestGuidanceMode(mock_missile)
        assert guidance.missile == mock_missile

        # Test that the implementation works
        yaw, pitch = guidance.compute(0.0, 0.0, mock_missile.position, Mock(), 0.1)
        assert yaw == 10.0
        assert pitch == 5.0
