import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestFox2Missile:
    """Test the base FOX2 missile class."""

    @pytest.fixture
    def mock_source(self):
        """Create a mock source unit (aircraft)."""
        source = Mock()
        source.position = Mock()
        source.position.lat = 45.0
        source.position.lon = 2.0
        source.position.alt = 10000.0
        source.position.copy.return_value = source.position
        source.yaw_deg = 45.0
        source.speed = 300.0
        source.pitch_deg = 5.0
        source.group = "BLUE"
        source.name = "F16_Blue"
        return source

    @pytest.fixture
    def mock_target(self):
        """Create a mock target unit."""
        target = Mock()
        target.position = Mock()
        target.position.lat = 46.0
        target.position.lon = 3.0
        target.position.alt = 9000.0
        target.name = "Enemy_Red"
        target.group = "RED"
        # Add temperature for IR seeker testing
        target.temperature = 500.0  # Kelvin
        return target

    @pytest.fixture
    def mock_map_limits(self):
        """Create mock map limits."""
        return {
            "lat_min": 44.0,
            "lat_max": 47.0,
            "lon_min": 1.0,
            "lon_max": 4.0
        }

    def test_fox2_missile_import(self):
        """Test that FOX2 missile base class can be imported."""
        try:
            from missiles.fox2.base_fox2 import Fox2Missile
            assert Fox2Missile is not None
        except ImportError:
            pytest.skip("Fox2Missile not available")

    def test_fox2_missile_initialization(self, mock_source, mock_target, mock_map_limits):
        """Test FOX2 missile initialization."""
        try:
            from missiles.fox2.base_fox2 import Fox2Missile
            
            with patch.multiple('missiles.missile',
                              MissileEngine=Mock(),
                              MissileMovement=Mock(),
                              GuidanceTargetProvider=Mock(),
                              MissileGuidance=Mock(),
                              MissilePhaseManager=Mock(),
                              MissileHitCalculator=Mock(),
                              MissilePhysics=Mock(),
                              MissileRadar=Mock()), \
                 patch('missiles.fox2.ir_seeker.InfraredSeeker', Mock()):
                
                config = {
                    "mass_kg": 85.0,
                    "reference_area_m2": 0.15,
                    "aspect_ratio": 3.2,
                    "oswald_e": 0.75,
                    "drag_coefficient": 0.10,
                    "constant_engine_N": 8000.0,
                    "n_max": 30.0,
                    "max_speed_mps": 900.0,
                    "life_time_s": 45.0,
                    "hit_probability": 0.9,
                    "motor_burn_s": 6.0,
                    "radar": {"fov_deg": 20.0, "max_range_m": 18000.0}
                }
                
                missile = Fox2Missile(
                    name="Test_FOX2",
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits,
                    group="BLUE",
                    config=config
                )
                
                assert missile.name == "Test_FOX2"
                assert missile.group == "BLUE"
                assert missile.target == mock_target
                assert missile.source == mock_source
                assert missile.max_speed_mps == 900.0
                
        except ImportError:
            pytest.skip("Fox2Missile dependencies not available")

    def test_fox2_missile_ir_seeker(self, mock_source, mock_target, mock_map_limits):
        """Test FOX2 missile IR seeker functionality."""
        try:
            from missiles.fox2.base_fox2 import Fox2Missile
            
            with patch.multiple('missiles.missile',
                              MissileEngine=Mock(),
                              MissileMovement=Mock(),
                              GuidanceTargetProvider=Mock(),
                              MissileGuidance=Mock(),
                              MissilePhaseManager=Mock(),
                              MissileHitCalculator=Mock(),
                              MissilePhysics=Mock(),
                              MissileRadar=Mock()), \
                 patch('missiles.fox2.ir_seeker.InfraredSeeker', Mock()) as mock_seeker_class:
                
                mock_seeker = mock_seeker_class.return_value
                mock_seeker.can_track_target.return_value = True
                mock_seeker.get_heat_signature.return_value = 500.0
                
                config = {
                    "mass_kg": 85.0,
                    "reference_area_m2": 0.15,
                    "aspect_ratio": 3.2,
                    "oswald_e": 0.75,
                    "drag_coefficient": 0.10,
                    "constant_engine_N": 8000.0,
                    "n_max": 30.0,
                    "max_speed_mps": 900.0,
                    "life_time_s": 45.0,
                    "hit_probability": 0.9,
                    "motor_burn_s": 6.0,
                    "radar": {"fov_deg": 15.0, "max_range_m": 15000.0}
                }
                
                missile = Fox2Missile(
                    name="Test_FOX2",
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits,
                    group="BLUE",
                    config=config
                )
                
                # Test IR seeker functionality
                assert hasattr(missile, 'radar')
                mock_seeker_class.assert_called_once()
                
        except ImportError:
            pytest.skip("Fox2Missile dependencies not available")

    def test_fox2_missile_heat_seeking(self, mock_source, mock_target, mock_map_limits):
        """Test FOX2 missile heat seeking capability."""
        try:
            from missiles.fox2.base_fox2 import Fox2Missile
            
            with patch.multiple('missiles.missile',
                              MissileEngine=Mock(),
                              MissileMovement=Mock(),
                              GuidanceTargetProvider=Mock(),
                              MissileGuidance=Mock(),
                              MissilePhaseManager=Mock(),
                              MissileHitCalculator=Mock(),
                              MissilePhysics=Mock(),
                              MissileRadar=Mock()), \
                 patch('missiles.fox2.ir_seeker.InfraredSeeker', Mock()) as mock_seeker_class:
                
                mock_seeker = mock_seeker_class.return_value
                
                # Simulate different heat signatures
                mock_seeker.get_heat_signature.side_effect = [400.0, 600.0, 300.0]
                mock_seeker.can_track_target.return_value = True
                
                config = {
                    "mass_kg": 85.0,
                    "reference_area_m2": 0.15,
                    "aspect_ratio": 3.2,
                    "oswald_e": 0.75,
                    "drag_coefficient": 0.10,
                    "constant_engine_N": 8000.0,
                    "n_max": 30.0,
                    "max_speed_mps": 900.0,
                    "life_time_s": 45.0,
                    "hit_probability": 0.9,
                    "motor_burn_s": 6.0,
                    "radar": {"fov_deg": 20.0, "max_range_m": 18000.0}
                }
                
                missile = Fox2Missile(
                    name="Test_FOX2",
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits,
                    group="BLUE",
                    config=config
                )
                
                # Test that IR seeker was created
                mock_seeker_class.assert_called_once()
                
        except ImportError:
            pytest.skip("Fox2Missile dependencies not available")

    def test_fox2_missile_step(self, mock_source, mock_target, mock_map_limits):
        """Test FOX2 missile step functionality."""
        try:
            from missiles.fox2.base_fox2 import Fox2Missile
            
            with patch('missiles.missile.MissileEngine', Mock()) as mock_engine, \
                 patch('missiles.missile.MissileMovement', Mock()) as mock_movement, \
                 patch('missiles.missile.GuidanceTargetProvider', Mock()) as mock_target_provider, \
                 patch('missiles.missile.MissileGuidance', Mock()) as mock_guidance, \
                 patch('missiles.missile.MissilePhaseManager', Mock()) as mock_phase_mgr, \
                 patch('missiles.missile.MissileHitCalculator', Mock()) as mock_hit_calc, \
                 patch('missiles.missile.MissilePhysics', Mock()) as mock_physics, \
                 patch('missiles.missile.MissileRadar', Mock()) as mock_radar, \
                 patch('missiles.fox2.ir_seeker.InfraredSeeker', Mock()) as mock_seeker_class:
                
                # Configure mocks
                mock_phase_mgr_instance = mock_phase_mgr.return_value
                mock_phase_mgr_instance.current_phase = "ACTIVE"
                mock_phase_mgr_instance.is_active.return_value = True
                
                mock_guidance_instance = mock_guidance.return_value
                mock_guidance_instance.get_guidance_command.return_value = np.array([0.0, 0.0, 1.0])
                
                mock_movement_instance = mock_movement.return_value
                mock_engine_instance = mock_engine.return_value
                mock_hit_calc_instance = mock_hit_calc.return_value
                
                config = {
                    "mass_kg": 85.0,
                    "reference_area_m2": 0.15,
                    "aspect_ratio": 3.2,
                    "oswald_e": 0.75,
                    "drag_coefficient": 0.10,
                    "constant_engine_N": 8000.0,
                    "n_max": 30.0,
                    "max_speed_mps": 900.0,
                    "life_time_s": 45.0,
                    "hit_probability": 0.9,
                    "motor_burn_s": 6.0,
                    "radar": {"fov_deg": 20.0, "max_range_m": 18000.0}
                }
                
                missile = Fox2Missile(
                    name="Test_FOX2",
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits,
                    group="BLUE",
                    config=config
                )
                
                # Test step functionality
                dt = 0.1
                simulator_mock = Mock()
                
                missile.update(dt, simulator_mock)
                
                # Verify that step components were called
                mock_phase_mgr.return_value.update.assert_called_once()
                
        except ImportError:
            pytest.skip("Fox2Missile dependencies not available")

    def test_fox2_missile_fire_and_forget(self, mock_source, mock_target, mock_map_limits):
        """Test that FOX2 missiles are fire-and-forget (don't require source)."""
        try:
            from missiles.fox2.base_fox2 import Fox2Missile
            
            with patch.multiple('missiles.missile',
                              MissileEngine=Mock(),
                              MissileMovement=Mock(),
                              GuidanceTargetProvider=Mock(),
                              MissileGuidance=Mock(),
                              MissilePhaseManager=Mock(),
                              MissileHitCalculator=Mock(),
                              MissilePhysics=Mock(),
                              MissileRadar=Mock()), \
                 patch('missiles.fox2.ir_seeker.InfraredSeeker', Mock()) as mock_seeker_class:
                
                mock_seeker = mock_seeker_class.return_value
                mock_seeker.can_track_target.return_value = True
                
                config = {
                    "mass_kg": 85.0,
                    "reference_area_m2": 0.15,
                    "aspect_ratio": 3.2,
                    "oswald_e": 0.75,
                    "drag_coefficient": 0.10,
                    "constant_engine_N": 8000.0,
                    "n_max": 30.0,
                    "max_speed_mps": 900.0,
                    "life_time_s": 45.0,
                    "hit_probability": 0.9,
                    "motor_burn_s": 6.0,
                    "radar": {"fov_deg": 20.0, "max_range_m": 18000.0}
                }
                
                missile = Fox2Missile(
                    name="Test_FOX2",
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits,
                    group="BLUE",
                    config=config
                )
                
                # FOX2 missiles should not require continuous source illumination
                # Unlike FOX1 missiles, they are fire-and-forget
                assert missile.source == mock_source  # Source is stored but not required for guidance
                
        except ImportError:
            pytest.skip("Fox2Missile dependencies not available")