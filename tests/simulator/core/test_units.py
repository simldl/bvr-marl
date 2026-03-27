"""
Comprehensive tests for simulator.core.units module.

Tests the Unit and FlyingUnit classes including positioning, movement,
velocity calculations, and physics integration.
"""

import math
from collections import namedtuple
from unittest.mock import MagicMock, Mock, patch

import pytest

from air_to_air_rl.simulator.core.helpers import Position
from air_to_air_rl.simulator.core.units import FlyingUnit, Unit, Velocity


class ConcreteUnit(Unit):
    """Concrete implementation of abstract Unit class for testing."""

    def update(self, tick_secs: float, sim) -> list:
        return []


class TestUnitBase:
    """Test the abstract Unit base class."""

    def test_unit_initialization(self):
        """Test basic unit initialization."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = ConcreteUnit(name="Test Unit", position=pos, yaw_deg=90.0, speed=200.0)

        assert unit.name == "Test Unit"
        assert unit.position == pos
        assert unit.yaw_deg == 90.0
        assert unit.speed == 200.0
        assert unit.id is None
        assert unit.desired_yaw_deg == 90.0

    def test_unit_initialization_with_angle_normalization(self):
        """Test that yaw angle is normalized during initialization."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)

        # Test positive angle > 360
        unit1 = ConcreteUnit(name="Unit1", position=pos, yaw_deg=450.0, speed=200.0)
        assert unit1.yaw_deg == pytest.approx(90.0, abs=1e-10)

        # Test negative angle
        unit2 = ConcreteUnit(name="Unit2", position=pos, yaw_deg=-90.0, speed=200.0)
        assert unit2.yaw_deg == pytest.approx(270.0, abs=1e-10)

    @patch("air_to_air_rl.simulator.core.units.normalize_angle")
    def test_angle_normalization_called(self, mock_normalize):
        """Test that normalize_angle is called during initialization."""
        mock_normalize.return_value = 45.0
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)

        unit = ConcreteUnit(name="Test", position=pos, yaw_deg=405.0, speed=200.0)

        mock_normalize.assert_called_once_with(405.0)
        assert unit.yaw_deg == 45.0
        assert unit.desired_yaw_deg == 45.0

    def test_abstract_update_method(self):
        """Test that Unit is abstract and requires update method implementation."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)

        # Cannot instantiate abstract Unit directly
        with pytest.raises(TypeError):
            Unit(name="Test", position=pos, yaw_deg=90.0, speed=200.0)

    def test_physics_step_default_implementation(self):
        """Test default physics_step implementation."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = ConcreteUnit(name="Test", position=pos, yaw_deg=90.0, speed=200.0)
        sim = Mock()

        result = unit.physics_step(1.0, sim)
        assert result is None

    def test_substep_update_delegates_to_physics_step(self):
        """Test that substep_update delegates to physics_step."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = ConcreteUnit(name="Test", position=pos, yaw_deg=90.0, speed=200.0)
        sim = Mock()

        # Mock physics_step to return a value
        unit.physics_step = Mock(return_value=["test_event"])

        result = unit.substep_update(0.5, sim)

        unit.physics_step.assert_called_once_with(0.5, sim)
        assert result == ["test_event"]

    def test_to_string_formatting(self):
        """Test string representation formatting."""
        pos = Position(lat=45.1234, lon=-123.5678, alt=1500.7)
        unit = ConcreteUnit(name="Fighter", position=pos, yaw_deg=180.5, speed=250.8)
        unit.id = 42

        result = unit.to_string()

        expected = "Fighter[42]: pos=(45.1234, -123.5678, 1500.7) h=180.5° s=250.8m/s"
        assert result == expected

    def test_to_string_without_id(self):
        """Test string representation when id is None."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = ConcreteUnit(name="Test", position=pos, yaw_deg=90.0, speed=200.0)

        result = unit.to_string()

        expected = "Test[None]: pos=(45.0000, -123.0000, 1000.0) h=90.0° s=200.0m/s"
        assert result == expected

    def test_unit_dataclass_properties(self):
        """Test that Unit behaves as a dataclass."""
        pos1 = Position(lat=45.0, lon=-123.0, alt=1000.0)
        pos2 = Position(lat=45.0, lon=-123.0, alt=1000.0)

        unit1 = ConcreteUnit(name="Test", position=pos1, yaw_deg=90.0, speed=200.0)
        unit2 = ConcreteUnit(name="Test", position=pos2, yaw_deg=90.0, speed=200.0)

        # Assign different IDs to make units different
        unit1.id = 1
        unit2.id = 2

        # With different IDs, units are different objects
        assert unit1 != unit2  # Different instances
        assert unit1.position == unit2.position  # But positions are equal


class TestVelocityNamedTuple:
    """Test the Velocity namedtuple."""

    def test_velocity_creation(self):
        """Test Velocity namedtuple creation."""
        vel = Velocity(10.0, 20.0, 5.0)

        assert vel.vx == 10.0
        assert vel.vy == 20.0
        assert vel.vz == 5.0

    def test_velocity_indexing(self):
        """Test Velocity indexing access."""
        vel = Velocity(-5.0, 15.0, -2.0)

        assert vel[0] == -5.0
        assert vel[1] == 15.0
        assert vel[2] == -2.0

    def test_velocity_immutability(self):
        """Test that Velocity is immutable."""
        vel = Velocity(1.0, 2.0, 3.0)

        with pytest.raises(AttributeError):
            vel.vx = 10.0  # Should be immutable

    def test_velocity_iteration(self):
        """Test Velocity iteration."""
        vel = Velocity(1.5, 2.5, 3.5)
        components = list(vel)

        assert components == [1.5, 2.5, 3.5]


class TestFlyingUnit:
    """Test the FlyingUnit class."""

    def test_flying_unit_initialization(self):
        """Test FlyingUnit initialization with default values."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = FlyingUnit(name="Fighter", position=pos, yaw_deg=90.0, speed=200.0)

        assert unit.name == "Fighter"
        assert unit.position == pos
        assert unit.yaw_deg == 90.0
        assert unit.speed == 200.0
        assert unit.pitch_deg == 0.0
        assert unit.roll_deg == 0.0
        assert unit.desired_pitch_deg == 0.0
        assert unit.desired_roll_deg == 0.0
        assert unit.desired_yaw_deg == 90.0

    def test_flying_unit_initialization_with_attitudes(self):
        """Test FlyingUnit initialization with pitch and roll."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = FlyingUnit(
            name="Bomber", position=pos, yaw_deg=180.0, speed=150.0, pitch_deg=5.0, roll_deg=-10.0
        )

        assert unit.pitch_deg == 5.0
        assert unit.roll_deg == -10.0
        assert unit.desired_pitch_deg == 5.0
        assert unit.desired_roll_deg == -10.0

    def test_flying_unit_post_init_calls_super(self):
        """Test that FlyingUnit.__post_init__ calls super().__post_init__."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)

        with patch.object(Unit, "__post_init__") as mock_super_post_init:
            FlyingUnit(name="Test", position=pos, yaw_deg=450.0, speed=200.0)
            mock_super_post_init.assert_called_once()

    def test_update_zero_speed_no_movement(self):
        """Test update with zero speed doesn't change position."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = FlyingUnit(name="Stationary", position=pos, yaw_deg=90.0, speed=0.0)
        sim = Mock()

        events = unit.update(1.0, sim)

        assert events == []
        assert unit.position.lat == 45.0
        assert unit.position.lon == -123.0
        assert unit.position.alt == 1000.0

    @patch("air_to_air_rl.simulator.core.units.geodetic_direct")
    def test_update_with_movement(self, mock_geodetic_direct):
        """Test update with movement calls geodetic_direct correctly."""
        mock_geodetic_direct.return_value = (45.1, -122.9, 1100.0)

        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = FlyingUnit(name="Moving", position=pos, yaw_deg=45.0, speed=100.0, pitch_deg=10.0)
        sim = Mock()

        events = unit.update(2.0, sim)

        # Calculate expected parameters
        distance = 100.0 * 2.0  # speed * tick_secs
        vertical = 100.0 * math.sin(math.radians(10.0)) * 2.0

        mock_geodetic_direct.assert_called_once_with(
            45.0, -123.0, 1000.0, 45.0, distance, vertical_distance=vertical
        )

        assert unit.position.lat == 45.1
        assert unit.position.lon == -122.9
        assert unit.position.alt == 1100.0
        assert events == []

    def test_update_negative_pitch(self):
        """Test update with negative pitch (descent)."""
        with patch("air_to_air_rl.simulator.core.units.geodetic_direct") as mock_geodetic_direct:
            mock_geodetic_direct.return_value = (45.1, -122.9, 900.0)

            pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
            unit = FlyingUnit(
                name="Diving", position=pos, yaw_deg=0.0, speed=100.0, pitch_deg=-10.0
            )
            sim = Mock()

            unit.update(1.0, sim)

            # Verify negative vertical component
            args = mock_geodetic_direct.call_args[1]
            assert args["vertical_distance"] < 0

    def test_physics_step_identical_to_update(self):
        """Test that physics_step behaves identically to update."""
        with patch("air_to_air_rl.simulator.core.units.geodetic_direct") as mock_geodetic_direct:
            mock_geodetic_direct.return_value = (45.05, -122.95, 1050.0)

            pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
            unit = FlyingUnit(name="Physics", position=pos, yaw_deg=90.0, speed=50.0, pitch_deg=5.0)
            sim = Mock()

            # Call both methods with same parameters
            events1 = unit.update(1.0, sim)

            # Reset position for second test but not the mock
            unit.position = Position(lat=45.0, lon=-123.0, alt=1000.0)

            events2 = unit.physics_step(1.0, sim)

            assert events1 == events2 == []
            # Both should have called geodetic_direct with identical parameters
            assert mock_geodetic_direct.call_count == 2
            call1_args = mock_geodetic_direct.call_args_list[0]
            call2_args = mock_geodetic_direct.call_args_list[1]
            assert call1_args == call2_args

    def test_velocity_property_calculation(self):
        """Test velocity property calculation with various attitudes."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = FlyingUnit(name="Test", position=pos, yaw_deg=0.0, speed=100.0, pitch_deg=0.0)

        vel = unit.velocity

        # North heading, level flight
        # velocity returns (vx=East, vy=North, vz=Up)
        assert vel.vx == pytest.approx(0.0, abs=1e-10)  # East component
        assert vel.vy == pytest.approx(100.0, abs=1e-10)  # North component
        assert vel.vz == pytest.approx(0.0, abs=1e-10)  # Up component

    def test_velocity_property_east_heading(self):
        """Test velocity with east heading."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = FlyingUnit(name="East", position=pos, yaw_deg=90.0, speed=100.0, pitch_deg=0.0)

        vel = unit.velocity

        # East heading, level flight
        # velocity returns (vx=East, vy=North, vz=Up)
        assert vel.vx == pytest.approx(100.0, abs=1e-10)  # East component
        assert vel.vy == pytest.approx(0.0, abs=1e-10)  # North component
        assert vel.vz == pytest.approx(0.0, abs=1e-10)  # Up component

    def test_velocity_property_with_pitch(self):
        """Test velocity calculation with pitch angle."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = FlyingUnit(name="Climbing", position=pos, yaw_deg=0.0, speed=100.0, pitch_deg=30.0)

        vel = unit.velocity

        expected_horizontal = 100.0 * math.cos(math.radians(30.0))
        expected_vertical = 100.0 * math.sin(math.radians(30.0))

        # North heading (yaw=0°), climbing (pitch=30°)
        # velocity returns (vx=East, vy=North, vz=Up)
        assert vel.vx == pytest.approx(0.0, abs=1e-10)  # East component (no east movement)
        assert vel.vy == pytest.approx(
            expected_horizontal, abs=1e-10
        )  # North component (horizontal velocity north)
        assert vel.vz == pytest.approx(
            expected_vertical, abs=1e-10
        )  # Up component (vertical velocity)

    def test_velocity_property_combined_yaw_pitch(self):
        """Test velocity with both yaw and pitch."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = FlyingUnit(name="Complex", position=pos, yaw_deg=45.0, speed=100.0, pitch_deg=20.0)

        vel = unit.velocity

        horizontal = 100.0 * math.cos(math.radians(20.0))
        vx_expected = horizontal * math.cos(math.radians(45.0))
        vy_expected = horizontal * math.sin(math.radians(45.0))
        vz_expected = 100.0 * math.sin(math.radians(20.0))

        assert vel.vx == pytest.approx(vx_expected, abs=1e-10)
        assert vel.vy == pytest.approx(vy_expected, abs=1e-10)
        assert vel.vz == pytest.approx(vz_expected, abs=1e-10)

    def test_velocity_property_negative_pitch(self):
        """Test velocity with negative pitch (diving)."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = FlyingUnit(name="Diving", position=pos, yaw_deg=180.0, speed=100.0, pitch_deg=-15.0)

        vel = unit.velocity

        horizontal = 100.0 * math.cos(math.radians(-15.0))
        vz_expected = 100.0 * math.sin(math.radians(-15.0))  # Negative (down)

        # South heading (yaw=180°), diving (pitch=-15°)
        # yaw_math = 90° - 180° = -90° (normalized to 270°)
        # velocity returns (vx=East, vy=North, vz=Up)
        # For South heading: vx (East) should be ~0, vy (North) should be negative (going south)
        yaw_math = 90.0 - 180.0  # -90° = 270° when normalized
        vx_expected = horizontal * math.cos(math.radians(yaw_math))  # East component
        vy_expected = horizontal * math.sin(
            math.radians(yaw_math)
        )  # North component (negative for south)

        assert vel.vx == pytest.approx(vx_expected, abs=1e-10)
        assert vel.vy == pytest.approx(vy_expected, abs=1e-10)
        assert vel.vz < 0  # Diving
        assert vel.vz == pytest.approx(vz_expected, abs=1e-10)

    def test_velocity_property_zero_speed(self):
        """Test velocity with zero speed."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = FlyingUnit(name="Stopped", position=pos, yaw_deg=90.0, speed=0.0, pitch_deg=45.0)

        vel = unit.velocity

        assert vel.vx == pytest.approx(0.0, abs=1e-10)
        assert vel.vy == pytest.approx(0.0, abs=1e-10)
        assert vel.vz == pytest.approx(0.0, abs=1e-10)

    def test_velocity_magnitude_consistency(self):
        """Test that velocity magnitude equals speed."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        speed = 150.0
        unit = FlyingUnit(name="Test", position=pos, yaw_deg=67.0, speed=speed, pitch_deg=13.0)

        vel = unit.velocity
        magnitude = math.sqrt(vel.vx**2 + vel.vy**2 + vel.vz**2)

        assert magnitude == pytest.approx(speed, abs=1e-10)


class TestFlyingUnitIntegration:
    """Integration tests for FlyingUnit with realistic scenarios."""

    @patch("air_to_air_rl.simulator.core.units.geodetic_direct")
    def test_realistic_flight_scenario(self, mock_geodetic_direct):
        """Test realistic flight scenario with multiple updates."""
        # Mock realistic position changes
        positions = [
            (45.1, -122.9, 1100.0),
            (45.2, -122.8, 1200.0),
            (45.3, -122.7, 1300.0),
        ]
        mock_geodetic_direct.side_effect = positions

        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        aircraft = FlyingUnit(
            name="Airliner", position=pos, yaw_deg=45.0, speed=250.0, pitch_deg=3.0
        )
        sim = Mock()

        # Multiple updates simulating flight progression
        for expected_pos in positions:
            events = aircraft.update(1.0, sim)
            assert events == []
            assert aircraft.position.lat == expected_pos[0]
            assert aircraft.position.lon == expected_pos[1]
            assert aircraft.position.alt == expected_pos[2]

    def test_different_update_intervals(self):
        """Test updates with different time intervals."""
        with patch("air_to_air_rl.simulator.core.units.geodetic_direct") as mock_geodetic_direct:
            mock_geodetic_direct.return_value = (45.1, -122.9, 1050.0)

            pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
            unit = FlyingUnit(
                name="Variable", position=pos, yaw_deg=0.0, speed=100.0, pitch_deg=5.0
            )
            sim = Mock()

            # Test different tick intervals
            intervals = [0.1, 0.5, 1.0, 2.0, 5.0]

            for dt in intervals:
                unit.position = Position(lat=45.0, lon=-123.0, alt=1000.0)  # Reset
                mock_geodetic_direct.reset_mock()

                unit.update(dt, sim)

                # Verify distance calculation scales with time
                call_args = mock_geodetic_direct.call_args
                distance_arg = call_args[0][4]  # Fifth positional argument
                expected_distance = 100.0 * dt
                assert distance_arg == pytest.approx(expected_distance, abs=1e-10)

    def test_extreme_attitudes(self):
        """Test with extreme pitch angles."""
        with patch("air_to_air_rl.simulator.core.units.geodetic_direct") as mock_geodetic_direct:
            mock_geodetic_direct.return_value = (45.0, -123.0, 2000.0)

            pos = Position(lat=45.0, lon=-123.0, alt=1000.0)

            # Test extreme climb
            unit1 = FlyingUnit(
                name="Rocket", position=pos.copy(), yaw_deg=0.0, speed=100.0, pitch_deg=89.0
            )
            unit1.update(1.0, Mock())

            # Test extreme dive
            unit2 = FlyingUnit(
                name="Dive", position=pos.copy(), yaw_deg=0.0, speed=100.0, pitch_deg=-89.0
            )
            unit2.update(1.0, Mock())

            # Both should complete without error
            assert mock_geodetic_direct.call_count == 2

    def test_high_speed_scenarios(self):
        """Test with high-speed scenarios."""
        with patch("air_to_air_rl.simulator.core.units.geodetic_direct") as mock_geodetic_direct:
            mock_geodetic_direct.return_value = (46.0, -121.0, 1000.0)

            pos = Position(lat=45.0, lon=-123.0, alt=1000.0)

            # Supersonic speed
            unit = FlyingUnit(name="Supersonic", position=pos, yaw_deg=90.0, speed=500.0)
            unit.update(1.0, Mock())

            # Verify large distance calculation
            call_args = mock_geodetic_direct.call_args
            distance_arg = call_args[0][4]
            assert distance_arg == 500.0  # 500 m/s * 1s

    def test_position_object_modification(self):
        """Test that position object is properly modified in-place."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = FlyingUnit(name="Test", position=pos, yaw_deg=0.0, speed=100.0)

        original_pos = unit.position

        with patch("air_to_air_rl.simulator.core.units.geodetic_direct") as mock_geodetic_direct:
            mock_geodetic_direct.return_value = (45.1, -122.9, 1100.0)

            unit.update(1.0, Mock())

            # Should modify the same Position object
            assert unit.position is original_pos
            assert unit.position.lat == 45.1
            assert unit.position.lon == -122.9
            assert unit.position.alt == 1100.0


class TestFlyingUnitEdgeCases:
    """Test edge cases and error conditions for FlyingUnit."""

    def test_nan_speed_handling(self):
        """Test behavior with NaN speed."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = FlyingUnit(name="NaN", position=pos, yaw_deg=0.0, speed=float("nan"))

        # Should not crash, but behavior with NaN is undefined
        # The speed > 0.0 check may fail with NaN
        events = unit.update(1.0, Mock())
        assert isinstance(events, list)

    def test_negative_speed_handling(self):
        """Test behavior with negative speed."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = FlyingUnit(name="Reverse", position=pos, yaw_deg=0.0, speed=-100.0)

        # Negative speed should not trigger movement (speed > 0.0 check)
        events = unit.update(1.0, Mock())
        assert events == []
        assert unit.position.lat == 45.0  # No change

    def test_very_small_speed(self):
        """Test behavior with very small positive speed."""
        with patch("air_to_air_rl.simulator.core.units.geodetic_direct") as mock_geodetic_direct:
            mock_geodetic_direct.return_value = (45.0, -123.0, 1000.0)

            pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
            unit = FlyingUnit(name="Slow", position=pos, yaw_deg=0.0, speed=1e-10)

            events = unit.update(1.0, Mock())

            # Should still trigger movement logic
            mock_geodetic_direct.assert_called_once()
            assert events == []

    @patch("air_to_air_rl.simulator.core.units.geodetic_direct")
    def test_geodetic_direct_exception_handling(self, mock_geodetic_direct):
        """Test behavior when geodetic_direct raises exception."""
        mock_geodetic_direct.side_effect = ValueError("Invalid coordinates")

        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
        unit = FlyingUnit(name="Error", position=pos, yaw_deg=0.0, speed=100.0)

        # Should propagate the exception
        with pytest.raises(ValueError):
            unit.update(1.0, Mock())

    def test_extreme_pitch_angles(self):
        """Test with extreme pitch angles beyond normal ranges."""
        pos = Position(lat=45.0, lon=-123.0, alt=1000.0)

        # Test pitch > 90 degrees
        unit1 = FlyingUnit(name="Inverted", position=pos, yaw_deg=0.0, speed=100.0, pitch_deg=120.0)
        vel1 = unit1.velocity

        # Should still compute velocity (mathematical validity)
        assert isinstance(vel1, Velocity)

        # Test very negative pitch
        unit2 = FlyingUnit(
            name="InvertedDive", position=pos, yaw_deg=0.0, speed=100.0, pitch_deg=-120.0
        )
        vel2 = unit2.velocity

        assert isinstance(vel2, Velocity)

    def test_multiple_consecutive_updates(self):
        """Test multiple consecutive updates maintain consistency."""
        positions = [(45.0, -123.0, 1000.0)]  # Starting position

        def side_effect(*args, **kwargs):
            # Mock function to simulate continuous movement
            lat, lon, alt = positions[-1]
            new_lat = lat + 0.01
            new_lon = lon + 0.01
            new_alt = alt + 10
            new_pos = (new_lat, new_lon, new_alt)
            positions.append(new_pos)
            return new_pos

        with patch("air_to_air_rl.simulator.core.units.geodetic_direct", side_effect=side_effect):
            pos = Position(lat=45.0, lon=-123.0, alt=1000.0)
            unit = FlyingUnit(name="Continuous", position=pos, yaw_deg=45.0, speed=100.0)
            sim = Mock()

            # Multiple updates
            for _ in range(5):
                events = unit.update(1.0, sim)
                assert events == []

            # Should have moved significantly
            assert unit.position.lat > 45.0
            assert unit.position.lon > -123.0
            assert unit.position.alt > 1000.0
