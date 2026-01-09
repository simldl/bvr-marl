import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass


class TestFox1Missile:
    """Test the base FOX1 missile class."""

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
        source.name = "F22_Blue"
        # Mock radar for Fox1 parent lock checking
        source.radar = Mock()
        source.radar.get_locked_targets.return_value = []  # Return empty list (iterable)
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
        return target

    @pytest.fixture
    def mock_map_limits(self):
        """Create mock map limits."""
        class MapLimits:
            def __init__(self):
                self.bottom_lat = 44.0
                self.top_lat = 47.0
                self.left_lon = 1.0
                self.right_lon = 4.0
                self.min_alt = 0.0
                self.max_alt = 20000.0
        return MapLimits()

    def test_fox1_missile_import(self):
        """Test that FOX1 missile base class can be imported."""
        try:
            from missiles.fox1.base_fox1 import Fox1Missile
            assert Fox1Missile is not None
        except ImportError:
            pytest.skip("Fox1Missile not available")

    def test_fox1_missile_initialization(self, mock_source, mock_target, mock_map_limits):
        """Test FOX1 missile initialization."""
        try:
            from missiles.fox1.base_fox1 import Fox1Missile
            
            with patch.multiple('missiles.missile',
                              MissileEngine=Mock(),
                              MissileMovement=Mock(),
                              GuidanceTargetProvider=Mock(),
                              MissileGuidance=Mock(),
                              MissilePhaseManager=Mock(),
                              MissileHitCalculator=Mock(),
                              MissilePhysics=Mock(),
                              MissileRadar=Mock()), \
                 patch('missiles.fox1.sarh_seeker.SemiActiveRadarSeeker', Mock()):
                
                config = {
                    "mass_kg": 200.0,
                    "reference_area_m2": 0.20,
                    "aspect_ratio": 3.0,
                    "oswald_e": 0.75,
                    "drag_coefficient": 0.12,
                    "constant_engine_N": 12000.0,
                    "n_max": 25.0,
                    "max_speed_mps": 1000.0,
                    "life_time_s": 60.0,
                    "hit_probability": 0.8,
                    "motor_burn_s": 10.0,
                    "radar": {"fov_deg": 30.0, "max_range_m": 50000.0}
                }
                
                missile = Fox1Missile(
                    name="Test_FOX1",
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits,
                    group="BLUE",
                    config=config
                )
                
                assert missile.name == "Test_FOX1"
                assert missile.group == "BLUE"
                assert missile.target == mock_target
                assert missile.source == mock_source
                assert missile.max_speed_mps == 1000.0
                
        except ImportError:
            pytest.skip("Fox1Missile dependencies not available")

    def test_fox1_missile_sarh_seeker(self, mock_source, mock_target, mock_map_limits):
        """Test FOX1 missile SARH seeker functionality."""
        try:
            from missiles.fox1.base_fox1 import Fox1Missile
            
            with patch.multiple('missiles.missile',
                              MissileEngine=Mock(),
                              MissileMovement=Mock(),
                              GuidanceTargetProvider=Mock(),
                              MissileGuidance=Mock(),
                              MissilePhaseManager=Mock(),
                              MissileHitCalculator=Mock(),
                              MissilePhysics=Mock(),
                              MissileRadar=Mock()) as mocks, \
                 patch('missiles.fox1.sarh_seeker.SemiActiveRadarSeeker', Mock()) as mock_seeker_class:
                
                mock_seeker = mock_seeker_class.return_value
                mock_seeker.can_track_target.return_value = True
                mock_seeker.lose_lock.return_value = None
                
                config = {
                    "mass_kg": 200.0,
                    "reference_area_m2": 0.20,
                    "aspect_ratio": 3.0,
                    "oswald_e": 0.75,
                    "drag_coefficient": 0.12,
                    "constant_engine_N": 12000.0,
                    "n_max": 25.0,
                    "max_speed_mps": 1000.0,
                    "life_time_s": 60.0,
                    "hit_probability": 0.8,
                    "motor_burn_s": 10.0,
                    "radar": {"fov_deg": 30.0, "max_range_m": 50000.0}
                }
                
                missile = Fox1Missile(
                    name="Test_FOX1",
                    firing_time_s=0.0,
                    target=mock_target,
                    source=mock_source,
                    map_limits=mock_map_limits,
                    group="BLUE",
                    config=config
                )
                
                # Test seeker functionality
                assert hasattr(missile, 'radar')
                mock_seeker_class.assert_called_once()
                
        except ImportError:
            pytest.skip("Fox1Missile dependencies not available")

    def test_fox1_missile_step(self, mock_source, mock_target, mock_map_limits):
        """Test FOX1 missile step functionality."""
        try:
            from missiles.fox1.base_fox1 import Fox1Missile

            with patch('missiles.missile.MissileEngine', Mock()) as mock_engine, \
                 patch('missiles.missile.MissileMovement', Mock()) as mock_movement, \
                 patch('missiles.missile.GuidanceTargetProvider', Mock()) as mock_target_provider, \
                 patch('missiles.missile.MissileGuidance', Mock()) as mock_guidance, \
                 patch('missiles.missile.MissilePhaseManager', Mock()) as mock_phase_mgr, \
                 patch('missiles.missile.MissileHitCalculator', Mock()) as mock_hit_calc, \
                 patch('missiles.missile.MissilePhysics', Mock()) as mock_physics, \
                 patch('missiles.missile.MissileRadar', Mock()) as mock_radar, \
                 patch('missiles.fox1.sarh_seeker.SemiActiveRadarSeeker', Mock()) as mock_seeker_class:

                # Configure mocks
                mock_phase_mgr_instance = mock_phase_mgr.return_value
                mock_phase_mgr_instance.current_phase = "ACTIVE"
                mock_phase_mgr_instance.is_active.return_value = True

                mock_guidance_instance = mock_guidance.return_value
                mock_guidance_instance.compute_guidance.return_value = (90.0, 15.0)

                mock_radar_instance = mock_radar.return_value
                mock_radar_instance.get_tracker_info.return_value = None
                mock_radar_instance.tracker_manager = Mock()

                mock_engine_instance = mock_engine.return_value
                mock_engine_instance.fuel_s = 10.0
                mock_engine_instance.should_remove_missile_on_energy.return_value = False

                config = {
                    "mass_kg": 200.0,
                    "reference_area_m2": 0.20,
                    "aspect_ratio": 3.0,
                    "oswald_e": 0.75,
                    "drag_coefficient": 0.12,
                    "constant_engine_N": 12000.0,
                    "n_max": 25.0,
                    "max_speed_mps": 1000.0,
                    "life_time_s": 60.0,
                    "hit_probability": 0.8,
                    "motor_burn_s": 10.0,
                    "radar": {
                        "horizontal_fov_deg": 60.0,
                        "vertical_fov_deg": 30.0,
                        "max_range_m": 50000.0,
                        "radar_frequency_hz": 10e9,
                        "tx_power_w": 5000.0,
                        "antenna_gain_db": 30.0,
                        "snr_threshold_db": 8.0,
                    }
                }

                missile = Fox1Missile(
                    name="Test_FOX1",
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
                simulator_mock.active_units = {}

                result = missile.update(dt, simulator_mock)

                # Verify that update returns a list
                assert isinstance(result, list)

                # Verify that step components were called
                mock_phase_mgr.return_value.update.assert_called_once()

        except ImportError:
            pytest.skip("Fox1Missile dependencies not available")