import pytest
import numpy as np
import math
from unittest.mock import Mock, patch


class TestLeadInterceptGuidance:
    """Test the lead intercept guidance system."""

    @pytest.fixture
    def mock_missile(self):
        """Create a mock missile."""
        missile = Mock()
        missile.speed = 800.0  # m/s
        missile.target_provider = Mock()
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

    def test_lead_intercept_import(self):
        """Test that lead intercept guidance can be imported."""
        try:
            from missiles.guidance.lead import LeadInterceptGuidance
            assert LeadInterceptGuidance is not None
        except ImportError:
            pytest.skip("LeadInterceptGuidance not available")

    def test_lead_intercept_initialization(self, mock_missile):
        """Test lead intercept guidance initialization."""
        try:
            from missiles.guidance.lead import LeadInterceptGuidance
            
            guidance = LeadInterceptGuidance(mock_missile)
            assert guidance.missile == mock_missile
            assert guidance.TGO_MIN == 0.2  # seconds
            assert guidance.TGO_MAX == 25.0  # seconds
            
        except ImportError:
            pytest.skip("LeadInterceptGuidance dependencies not available")

    def test_lead_intercept_no_target_velocity(self, mock_missile, mock_missile_position, mock_target_position):
        """Test lead intercept fallback to direct pursuit when no target velocity."""
        try:
            from missiles.guidance.lead import LeadInterceptGuidance
            
            with patch('missiles.guidance.lead.geodetic_to_enu') as mock_enu, \
                 patch('missiles.guidance.lead.DirectPursuitGuidance') as mock_direct_class:
                
                mock_enu.return_value = [1000.0, 2000.0, 500.0]  # ENU vector
                
                # No target velocity available
                mock_missile.target_provider.get_guidance_velocity.return_value = None
                
                # Mock direct pursuit guidance
                mock_direct_instance = Mock()
                mock_direct_instance.compute.return_value = (45.0, 10.0)
                mock_direct_class.return_value = mock_direct_instance
                
                guidance = LeadInterceptGuidance(mock_missile)
                
                yaw, pitch = guidance.compute(
                    current_yaw_deg=0.0,
                    current_pitch_deg=0.0,
                    missile_position=mock_missile_position,
                    target_position=mock_target_position,
                    tick_secs=0.1
                )
                
                # Should fall back to direct pursuit
                assert yaw == 45.0
                assert pitch == 10.0
                
                mock_direct_instance.compute.assert_called_once()
                
        except ImportError:
            pytest.skip("LeadInterceptGuidance dependencies not available")

    def test_lead_intercept_with_target_velocity(self, mock_missile, mock_missile_position, mock_target_position):
        """Test lead intercept calculation with target velocity."""
        try:
            from missiles.guidance.lead import LeadInterceptGuidance
            
            with patch('missiles.guidance.lead.geodetic_to_enu') as mock_enu:
                # Mock relative position: target 1000m east, 2000m north, 500m up
                mock_enu.return_value = [1000.0, 2000.0, 500.0]
                
                # Mock target velocity: 200 m/s north
                mock_missile.target_provider.get_guidance_velocity.return_value = [0.0, 200.0, 0.0]
                mock_missile.speed = 900.0  # m/s (faster than target)
                
                guidance = LeadInterceptGuidance(mock_missile)
                
                yaw, pitch = guidance.compute(
                    current_yaw_deg=0.0,
                    current_pitch_deg=0.0,
                    missile_position=mock_missile_position,
                    target_position=mock_target_position,
                    tick_secs=0.1
                )
                
                # Should calculate intercept point and aim there
                assert 0.0 <= yaw < 360.0
                assert -89.0 <= pitch <= 89.0
                
                # With target moving north and missile starting from south,
                # should aim somewhat north of the current target position
                
        except ImportError:
            pytest.skip("LeadInterceptGuidance dependencies not available")

    def test_lead_intercept_impossible_geometry(self, mock_missile, mock_missile_position, mock_target_position):
        """Test lead intercept when geometry is impossible (target too fast)."""
        try:
            from missiles.guidance.lead import LeadInterceptGuidance
            
            with patch('missiles.guidance.lead.geodetic_to_enu') as mock_enu, \
                 patch('missiles.guidance.lead.DirectPursuitGuidance') as mock_direct_class:
                
                mock_enu.return_value = [5000.0, 0.0, 0.0]  # Target 5km east
                
                # Target moving very fast (faster than missile can catch)
                mock_missile.target_provider.get_guidance_velocity.return_value = [1200.0, 0.0, 0.0]
                mock_missile.speed = 800.0  # m/s (slower than target)
                
                # Mock direct pursuit fallback
                mock_direct_instance = Mock()
                mock_direct_instance.compute.return_value = (90.0, 0.0)
                mock_direct_class.return_value = mock_direct_instance
                
                guidance = LeadInterceptGuidance(mock_missile)
                
                yaw, pitch = guidance.compute(
                    current_yaw_deg=0.0,
                    current_pitch_deg=0.0,
                    missile_position=mock_missile_position,
                    target_position=mock_target_position,
                    tick_secs=0.1
                )
                
                # Should fall back to direct pursuit
                assert yaw == 90.0
                assert pitch == 0.0
                
                mock_direct_instance.compute.assert_called_once()
                
        except ImportError:
            pytest.skip("LeadInterceptGuidance dependencies not available")

    def test_lead_intercept_time_clamping(self, mock_missile, mock_missile_position, mock_target_position):
        """Test that intercept time is clamped to reasonable bounds."""
        try:
            from missiles.guidance.lead import LeadInterceptGuidance
            
            with patch('missiles.guidance.lead.geodetic_to_enu') as mock_enu:
                mock_enu.return_value = [100.0, 0.0, 0.0]  # Very close target
                
                # Very slow target velocity
                mock_missile.target_provider.get_guidance_velocity.return_value = [1.0, 0.0, 0.0]
                mock_missile.speed = 800.0
                
                guidance = LeadInterceptGuidance(mock_missile)
                
                yaw, pitch = guidance.compute(
                    current_yaw_deg=0.0,
                    current_pitch_deg=0.0,
                    missile_position=mock_missile_position,
                    target_position=mock_target_position,
                    tick_secs=0.1
                )
                
                # Should produce reasonable angles even with very short intercept time
                assert 0.0 <= yaw < 360.0
                assert -89.0 <= pitch <= 89.0
                
        except ImportError:
            pytest.skip("LeadInterceptGuidance dependencies not available")

    def test_lead_intercept_inheritance(self, mock_missile):
        """Test that lead intercept inherits from BaseGuidanceMode."""
        try:
            from missiles.guidance.lead import LeadInterceptGuidance
            from missiles.guidance.base import BaseGuidanceMode
            
            guidance = LeadInterceptGuidance(mock_missile)
            assert isinstance(guidance, BaseGuidanceMode)
            
        except ImportError:
            pytest.skip("LeadInterceptGuidance dependencies not available")

    def test_lead_intercept_pitch_clamping(self, mock_missile, mock_missile_position, mock_target_position):
        """Test that pitch is clamped to valid range."""
        try:
            from missiles.guidance.lead import LeadInterceptGuidance
            
            with patch('missiles.guidance.lead.geodetic_to_enu') as mock_enu:
                # Mock target very high above missile
                mock_enu.return_value = [100.0, 0.0, 10000.0]  # High altitude difference
                
                mock_missile.target_provider.get_guidance_velocity.return_value = [0.0, 0.0, 10.0]
                mock_missile.speed = 800.0
                
                guidance = LeadInterceptGuidance(mock_missile)
                
                yaw, pitch = guidance.compute(
                    current_yaw_deg=0.0,
                    current_pitch_deg=0.0,
                    missile_position=mock_missile_position,
                    target_position=mock_target_position,
                    tick_secs=0.1
                )
                
                # Pitch should be clamped to [-89, 89] degrees
                assert -89.0 <= pitch <= 89.0
                
        except ImportError:
            pytest.skip("LeadInterceptGuidance dependencies not available")

    def test_lead_intercept_invalid_target_velocity(self, mock_missile, mock_missile_position, mock_target_position):
        """Test lead intercept with invalid target velocity."""
        try:
            from missiles.guidance.lead import LeadInterceptGuidance
            
            with patch('missiles.guidance.lead.geodetic_to_enu') as mock_enu, \
                 patch('missiles.guidance.lead.DirectPursuitGuidance') as mock_direct_class:
                
                mock_enu.return_value = [1000.0, 1000.0, 0.0]
                
                # Invalid target velocity (contains NaN)
                mock_missile.target_provider.get_guidance_velocity.return_value = [100.0, float('nan'), 0.0]
                
                # Mock direct pursuit fallback
                mock_direct_instance = Mock()
                mock_direct_instance.compute.return_value = (45.0, 0.0)
                mock_direct_class.return_value = mock_direct_instance
                
                guidance = LeadInterceptGuidance(mock_missile)
                
                yaw, pitch = guidance.compute(
                    current_yaw_deg=0.0,
                    current_pitch_deg=0.0,
                    missile_position=mock_missile_position,
                    target_position=mock_target_position,
                    tick_secs=0.1
                )
                
                # Should fall back to direct pursuit
                assert yaw == 45.0
                assert pitch == 0.0
                
        except ImportError:
            pytest.skip("LeadInterceptGuidance dependencies not available")

    def test_lead_intercept_linear_case(self, mock_missile, mock_missile_position, mock_target_position):
        """Test lead intercept with linear equation case (a ≈ 0)."""
        try:
            from missiles.guidance.lead import LeadInterceptGuidance
            
            with patch('missiles.guidance.lead.geodetic_to_enu') as mock_enu:
                mock_enu.return_value = [1000.0, 0.0, 0.0]  # Target 1km east
                
                # Target velocity same magnitude as missile speed (creates linear case)
                mock_missile.target_provider.get_guidance_velocity.return_value = [800.0, 0.0, 0.0]
                mock_missile.speed = 800.0
                
                guidance = LeadInterceptGuidance(mock_missile)
                
                yaw, pitch = guidance.compute(
                    current_yaw_deg=0.0,
                    current_pitch_deg=0.0,
                    missile_position=mock_missile_position,
                    target_position=mock_target_position,
                    tick_secs=0.1
                )
                
                # Should handle linear equation case
                assert 0.0 <= yaw < 360.0
                assert -89.0 <= pitch <= 89.0
                
        except ImportError:
            pytest.skip("LeadInterceptGuidance dependencies not available")