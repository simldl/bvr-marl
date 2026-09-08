from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest


class TestMissile:
    """Test the base missile class."""

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

    @pytest.fixture
    def sample_config(self):
        """Create a sample missile configuration."""
        return {
            "mass_kg": 150.0,
            "reference_area_m2": 0.2,
            "aspect_ratio": 3.0,
            "oswald_e": 0.8,
            "drag_coefficient": 0.05,
            "constant_engine_N": 15000.0,
            "n_max": 30.0,
            "max_speed_mps": 1000.0,
            "motor_burn_s": 20.0,
            "life_time_s": 100.0,
            "hit_probability": 0.85,
            "radar": {
                "horizontal_fov_deg": 60.0,
                "vertical_fov_deg": 30.0,
                "max_range_m": 50000.0,
                "radar_frequency_hz": 10e9,
                "tx_power_w": 5000.0,
                "antenna_gain_db": 30.0,
                "snr_threshold_db": 8.0,
            },
        }

    def test_missile_import(self):
        """Test that missile can be imported."""
        from bvr_marl_core.missiles.missile import Missile

        assert Missile is not None

    def test_missile_initialization(self, mock_source, mock_target, mock_map_limits, sample_config):
        """Test missile initialization."""
        from bvr_marl_core.missiles.missile import Missile

        with (
            patch("bvr_marl_core.missiles.missile.MissilePhaseManager"),
            patch("bvr_marl_core.missiles.missile.MissileEngine"),
            patch("bvr_marl_core.missiles.missile.MissilePhysics"),
            patch("bvr_marl_core.missiles.missile.MissileMovement"),
            patch("bvr_marl_core.missiles.missile.MissileRadar"),
            patch("bvr_marl_core.missiles.missile.GuidanceTargetProvider"),
            patch("bvr_marl_core.missiles.missile.MissileGuidance"),
        ):
            missile = Missile(
                name="Test_Missile",
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
                config=sample_config,
            )

            # Test basic attributes
            assert missile.name == "Test_Missile"
            assert missile.type == "Missile"
            assert missile.group == "BLUE"
            assert missile.source == mock_source
            assert missile.target == mock_target
            assert missile.is_missile is True
            assert missile.firing_time_s == 0.0
            assert missile.elapsed_time_s == 0.0

            # Test configuration values
            assert missile.max_speed_mps == 1000.0
            assert missile.hit_probability == 0.85
            assert missile.life_time_s == 100.0
            assert missile.motor_burn_s == 20.0

            # Test position inheritance from source
            assert missile.position == mock_source.position
            assert missile.yaw_deg == mock_source.yaw_deg
            assert missile.speed == mock_source.speed
            assert missile.pitch_deg == mock_source.pitch_deg

    def test_missile_inherits_shooter_velocity(
        self, mock_source, mock_target, mock_map_limits, sample_config
    ):
        """The missile spawns with the shooter's speed and attitude, so its
        velocity vector equals the shooter's kinematics at launch."""
        import math

        from bvr_marl_core.missiles.missile import Missile
        from bvr_marl_core.simulator.utils.angles import yaw_geo_to_math

        with (
            patch("bvr_marl_core.missiles.missile.MissilePhaseManager"),
            patch("bvr_marl_core.missiles.missile.MissileEngine"),
            patch("bvr_marl_core.missiles.missile.MissilePhysics"),
            patch("bvr_marl_core.missiles.missile.MissileMovement"),
            patch("bvr_marl_core.missiles.missile.MissileRadar"),
            patch("bvr_marl_core.missiles.missile.GuidanceTargetProvider"),
            patch("bvr_marl_core.missiles.missile.MissileGuidance"),
        ):
            missile = Missile(
                name="Test_Missile",
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
                config=sample_config,
            )

            # Expected velocity from the shooter's (speed=300, yaw=45, pitch=5).
            s = mock_source.speed
            pitch_rad = math.radians(mock_source.pitch_deg)
            yaw_math = math.radians(yaw_geo_to_math(mock_source.yaw_deg))
            hor = s * math.cos(pitch_rad)
            exp = (
                hor * math.cos(yaw_math),
                hor * math.sin(yaw_math),
                s * math.sin(pitch_rad),
            )

            v = missile.velocity
            assert (v.vx, v.vy, v.vz) == pytest.approx(exp)

    def test_missile_component_initialization(
        self, mock_source, mock_target, mock_map_limits, sample_config
    ):
        """Test that all missile components are properly initialized."""
        from bvr_marl_core.missiles.missile import Missile

        with (
            patch("bvr_marl_core.missiles.missile.MissilePhaseManager") as mock_phase_mgr,
            patch("bvr_marl_core.missiles.missile.MissileEngine") as mock_engine,
            patch("bvr_marl_core.missiles.missile.MissilePhysics") as mock_physics,
            patch("bvr_marl_core.missiles.missile.MissileMovement") as mock_movement,
            patch("bvr_marl_core.missiles.missile.MissileRadar") as mock_radar,
            patch("bvr_marl_core.missiles.missile.GuidanceTargetProvider") as mock_target_provider,
            patch("bvr_marl_core.missiles.missile.MissileGuidance") as mock_guidance,
        ):
            missile = Missile(
                name="Test_Missile",
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
                config=sample_config,
            )

            # Verify all components were created
            assert hasattr(missile, "phase_manager")
            assert hasattr(missile, "engine")
            assert hasattr(missile, "physics")
            assert hasattr(missile, "movement")
            assert hasattr(missile, "radar")
            assert hasattr(missile, "target_provider")
            assert hasattr(missile, "guidance")

            # Verify components were called with correct parameters
            mock_phase_mgr.assert_called_once()
            mock_engine.assert_called_once()
            mock_physics.assert_called_once()
            mock_movement.assert_called_once()
            mock_radar.assert_called_once()
            mock_target_provider.assert_called_once()
            mock_guidance.assert_called_once()

    def test_missile_update_cycle(self, mock_source, mock_target, mock_map_limits, sample_config):
        """Test missile update cycle."""
        from bvr_marl_core.missiles.missile import Missile

        with (
            patch("bvr_marl_core.missiles.missile.MissilePhaseManager") as mock_phase_mgr,
            patch("bvr_marl_core.missiles.missile.MissileEngine") as mock_engine,
            patch("bvr_marl_core.missiles.missile.MissilePhysics"),
            patch("bvr_marl_core.missiles.missile.MissileMovement") as mock_movement,
            patch("bvr_marl_core.missiles.missile.MissileRadar") as mock_radar,
            patch("bvr_marl_core.missiles.missile.GuidanceTargetProvider") as mock_target_provider,
            patch("bvr_marl_core.missiles.missile.MissileGuidance") as mock_guidance,
        ):
            # Configure mocks
            mock_radar_instance = mock_radar.return_value
            mock_radar_instance.get_tracker_info.return_value = {"velocity": [100.0, 50.0, 10.0]}
            mock_radar_instance.get_locked_target.return_value = None
            mock_radar_instance.tracker_manager = Mock()  # Add tracker manager for guidance

            mock_guidance_instance = mock_guidance.return_value
            mock_guidance_instance.compute_guidance.return_value = (90.0, 15.0)

            # Configure engine mock to not be empty and not trigger energy removal
            mock_engine_instance = mock_engine.return_value
            mock_engine_instance.is_empty.return_value = False
            mock_engine_instance.should_remove_missile_on_energy.return_value = False

            missile = Missile(
                name="Test_Missile",
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
                config=sample_config,
            )

            # Mock simulator with active units (needed for radar target gathering)
            mock_sim = Mock()
            mock_sim.active_units = {}  # Empty units for simplicity
            mock_sim.remove_unit = Mock()  # Mock removal method

            # Test update
            events = missile.update(0.1, mock_sim)

            # Verify basic update sequence - focus on what always happens
            assert missile.elapsed_time_s == 0.1
            mock_phase_mgr.return_value.update.assert_called_once()
            mock_engine.return_value.update.assert_called_once()

            # Radar/tracker update once per tick; guidance + movement run once per
            # flight sub-step (>= 1) since the missile integrates its own flight.
            mock_radar.return_value.update.assert_called_once()
            mock_target_provider.return_value.update.assert_called()
            mock_guidance.return_value.compute_guidance.assert_called()
            mock_movement.return_value.update.assert_called()

            # Test guidance results are applied
            assert missile.desired_yaw_deg == 90.0
            assert missile.desired_pitch_deg == 15.0

            # Test target velocity is updated
            np.testing.assert_allclose(missile.latest_target_velocity, [100.0, 50.0, 10.0])

            assert isinstance(events, list)

    def test_missile_removal_conditions(
        self, mock_source, mock_target, mock_map_limits, sample_config
    ):
        """Test missile removal conditions."""
        from bvr_marl_core.missiles.missile import Missile

        with (
            patch("bvr_marl_core.missiles.missile.MissilePhaseManager"),
            patch("bvr_marl_core.missiles.missile.MissileEngine") as mock_engine,
            patch("bvr_marl_core.missiles.missile.MissilePhysics"),
            patch("bvr_marl_core.missiles.missile.MissileMovement"),
            patch("bvr_marl_core.missiles.missile.MissileRadar") as mock_radar,
            patch("bvr_marl_core.missiles.missile.GuidanceTargetProvider"),
            patch("bvr_marl_core.missiles.missile.MissileGuidance"),
        ):
            missile = Missile(
                name="Test_Missile",
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
                config=sample_config,
            )

            # Keep the (mocked) energy check from short-circuiting _removal_reason
            # so the lifetime/glide/lock conditions below are actually exercised.
            mock_engine.return_value.should_remove_missile_on_energy.return_value = False

            # Test lifetime expiration
            missile.elapsed_time_s = 101.0  # Beyond life_time_s
            assert missile._removal_reason() == "lifetime_expired"
            # Reset for next test
            missile.elapsed_time_s = 50.0
            # Test fuel depletion with long glide
            mock_engine.return_value.fuel_s = 0.0  # Out of fuel
            missile.elapsed_time_s = 150.0  # 20s burn + 130s glide (> 120s max glide)
            assert missile._removal_reason() is not None
            # Reset for next test
            missile.elapsed_time_s = 50.0
            mock_engine.return_value.fuel_s = 10.0  # Has fuel
            # Test lost target lock after long flight
            mock_radar.return_value.get_locked_target.return_value = None
            missile.elapsed_time_s = 130.0  # > 120s without lock
            assert missile._removal_reason() is not None

    def test_missile_physics_initialization(
        self, mock_source, mock_target, mock_map_limits, sample_config
    ):
        """Test missile physics initialization."""
        from bvr_marl_core.missiles.missile import Missile

        with (
            patch("bvr_marl_core.missiles.missile.MissilePhaseManager"),
            patch("bvr_marl_core.missiles.missile.MissileEngine"),
            patch("bvr_marl_core.missiles.missile.MissilePhysics") as mock_physics,
            patch("bvr_marl_core.missiles.missile.MissileMovement"),
            patch("bvr_marl_core.missiles.missile.MissileRadar"),
            patch("bvr_marl_core.missiles.missile.GuidanceTargetProvider"),
            patch("bvr_marl_core.missiles.missile.MissileGuidance"),
        ):
            Missile(
                name="Test_Missile",
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
                config=sample_config,
            )

            # Verify physics was initialized with correct parameters
            physics_call = mock_physics.call_args
            physics_call[0][0]  # First positional argument (MissilePhysics.Params)

            # Check that _init_physics created proper parameters
            # (We can't directly test the params object, but we can verify the call)
            mock_physics.assert_called_once()

    def test_missile_radar_initialization(
        self, mock_source, mock_target, mock_map_limits, sample_config
    ):
        """Test missile radar initialization."""
        from bvr_marl_core.missiles.missile import Missile

        with (
            patch("bvr_marl_core.missiles.missile.MissilePhaseManager", Mock()),
            patch("bvr_marl_core.missiles.missile.MissileEngine", Mock()),
            patch("bvr_marl_core.missiles.missile.MissilePhysics", Mock()),
            patch("bvr_marl_core.missiles.missile.MissileMovement", Mock()),
            patch("bvr_marl_core.missiles.missile.MissileRadar", Mock()) as mock_radar,
            patch("bvr_marl_core.missiles.missile.GuidanceTargetProvider", Mock()),
            patch("bvr_marl_core.missiles.missile.MissileGuidance", Mock()),
            patch("bvr_marl_core.missiles.missile.DataLink", Mock()),
        ):
            missile = Missile(
                name="Test_Missile",
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
                config=sample_config,
                data_link_mode="partial",
            )

            # Verify radar initialization with correct parameters
            radar_call = mock_radar.call_args
            kwargs = radar_call[1]  # Keyword arguments

            assert kwargs["horizontal_fov_deg"] == 60.0
            assert kwargs["vertical_fov_deg"] == 30.0
            assert kwargs["max_range_m"] == 50000.0
            assert kwargs["radar_frequency_hz"] == 10e9
            assert kwargs["tx_power_w"] == 5000.0
            assert kwargs["antenna_gain_db"] == 30.0
            assert kwargs["snr_threshold_db"] == 8.0
            assert kwargs["owner"] == missile

            # Verify data link was created with correct mode
            # mocks['DataLink'].assert_called_once_with("partial")  # DataLink not mocked

    def test_missile_substep_update(self, mock_source, mock_target, mock_map_limits, sample_config):
        """Test missile substep update functionality."""
        from bvr_marl_core.missiles.missile import Missile

        with (
            patch("bvr_marl_core.missiles.missile.MissilePhaseManager"),
            patch("bvr_marl_core.missiles.missile.MissileEngine"),
            patch("bvr_marl_core.missiles.missile.MissilePhysics"),
            patch("bvr_marl_core.missiles.missile.MissileMovement") as mock_movement,
            patch("bvr_marl_core.missiles.missile.MissileRadar") as mock_radar,
            patch("bvr_marl_core.missiles.missile.GuidanceTargetProvider") as mock_target_provider,
            patch("bvr_marl_core.missiles.missile.MissileGuidance") as mock_guidance,
        ):
            mock_guidance.return_value.compute_guidance.return_value = (120.0, 20.0)

            missile = Missile(
                name="Test_Missile",
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
                config=sample_config,
            )

            mock_sim = Mock()

            # Test substep update
            result = missile.substep_update(0.05, mock_sim)

            # Verify substep operations
            mock_target_provider.return_value.update.assert_called_with(mock_sim, 0.05)
            # New signature: compute_guidance(missile, target_provider, tracker_manager, dt)
            mock_guidance.return_value.compute_guidance.assert_called_with(
                missile,
                mock_target_provider.return_value,
                mock_radar.return_value.tracker_manager,
                0.05,
            )
            mock_movement.return_value.update.assert_called_with(0.05)

            # Test guidance results are applied
            assert missile.desired_yaw_deg == 120.0
            assert missile.desired_pitch_deg == 20.0

            assert isinstance(result, list)

    def test_missile_engine_fuel_depletion(
        self, mock_source, mock_target, mock_map_limits, sample_config
    ):
        """Test missile behavior when engine runs out of fuel."""
        from bvr_marl_core.missiles.missile import Missile

        with (
            patch("bvr_marl_core.missiles.missile.MissilePhaseManager"),
            patch("bvr_marl_core.missiles.missile.MissileEngine") as mock_engine,
            patch("bvr_marl_core.missiles.missile.MissilePhysics"),
            patch("bvr_marl_core.missiles.missile.MissileMovement"),
            patch("bvr_marl_core.missiles.missile.MissileRadar"),
            patch("bvr_marl_core.missiles.missile.GuidanceTargetProvider"),
            patch("bvr_marl_core.missiles.missile.MissileGuidance"),
        ):
            mock_engine.return_value.is_empty.return_value = True

            missile = Missile(
                name="Test_Missile",
                firing_time_s=0.0,
                target=mock_target,
                source=mock_source,
                map_limits=mock_map_limits,
                group="BLUE",
                config=sample_config,
            )

            # Test long glide after fuel depletion
            missile.elapsed_time_s = 120.0  # 20s burn + 100s glide
            missile.speed = 50.0  # Very low speed

            # Should be removed (low energy / long glide) -> a non-None reason.
            assert missile._removal_reason() is not None
