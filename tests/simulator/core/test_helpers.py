#!/usr/bin/env python3
"""
Comprehensive tests for simulator.core.helpers module.
Tests Position class and utility functions for unit distance and bearing calculations.
"""

import math

import pytest

from air_to_air_rl.simulator.core.helpers import Position, clamp, units_bearing, units_distance_km


class TestPosition:
    """Test Position dataclass functionality."""

    def test_position_init(self):
        """Test Position initialization."""
        pos = Position(52.0, 8.0, 1000.0)

        assert pos.lat == 52.0
        assert pos.lon == 8.0
        assert pos.alt == 1000.0

    def test_position_init_with_floats(self):
        """Test Position with various float values."""
        pos = Position(52.123456, 8.987654, 1234.5)

        assert pos.lat == 52.123456
        assert pos.lon == 8.987654
        assert pos.alt == 1234.5

    def test_position_init_with_negative_values(self):
        """Test Position with negative coordinates."""
        pos = Position(-45.0, -120.0, -100.0)

        assert pos.lat == -45.0
        assert pos.lon == -120.0
        assert pos.alt == -100.0

    def test_position_init_with_zero_values(self):
        """Test Position with zero coordinates."""
        pos = Position(0.0, 0.0, 0.0)

        assert pos.lat == 0.0
        assert pos.lon == 0.0
        assert pos.alt == 0.0

    def test_position_init_with_integers(self):
        """Test Position with integer values (should be converted to float)."""
        pos = Position(52, 8, 1000)

        assert pos.lat == 52.0
        assert pos.lon == 8.0
        assert pos.alt == 1000.0

    def test_position_copy(self):
        """Test Position copy functionality."""
        original = Position(51.0, 8.0, 700.0)
        copied = original.copy()

        # Should be equal but not the same object
        assert copied.lat == original.lat
        assert copied.lon == original.lon
        assert copied.alt == original.alt
        assert copied is not original
        assert copied == original  # dataclass equality

    def test_position_copy_independence(self):
        """Test that copied Position is independent of original."""
        original = Position(51.0, 8.0, 700.0)
        copied = original.copy()

        # Modify original
        original.lat = 52.0
        original.lon = 9.0
        original.alt = 800.0

        # Copy should be unchanged
        assert copied.lat == 51.0
        assert copied.lon == 8.0
        assert copied.alt == 700.0

    def test_position_equality(self):
        """Test Position equality comparison."""
        pos1 = Position(52.0, 8.0, 1000.0)
        pos2 = Position(52.0, 8.0, 1000.0)
        pos3 = Position(52.1, 8.0, 1000.0)

        assert pos1 == pos2
        assert pos1 != pos3
        assert pos2 != pos3

    def test_position_attributes_mutable(self):
        """Test that Position attributes can be modified."""
        pos = Position(52.0, 8.0, 1000.0)

        pos.lat = 53.0
        pos.lon = 9.0
        pos.alt = 1500.0

        assert pos.lat == 53.0
        assert pos.lon == 9.0
        assert pos.alt == 1500.0

    def test_position_edge_cases(self):
        """Test Position with edge case values."""
        # Very large values
        pos_large = Position(90.0, 180.0, 50000.0)
        assert pos_large.lat == 90.0
        assert pos_large.lon == 180.0
        assert pos_large.alt == 50000.0

        # Very small values
        pos_small = Position(-90.0, -180.0, 0.001)
        assert pos_small.lat == -90.0
        assert pos_small.lon == -180.0
        assert pos_small.alt == 0.001

        # Float precision
        pos_precise = Position(52.123456789, 8.987654321, 1000.123456)
        assert pos_precise.lat == 52.123456789
        assert pos_precise.lon == 8.987654321
        assert pos_precise.alt == 1000.123456


class MockUnit:
    """Mock unit class for testing distance and bearing functions."""

    def __init__(self, lat, lon, alt):
        self.position = Position(lat, lon, alt)


class TestUnitsDistanceKm:
    """Test units_distance_km function."""

    def test_units_distance_same_position(self):
        """Test distance between units at same position."""
        unit_a = MockUnit(52.0, 8.0, 1000.0)
        unit_b = MockUnit(52.0, 8.0, 1000.0)

        distance = units_distance_km(unit_a, unit_b)

        assert distance == 0.0

    def test_units_distance_horizontal_only(self):
        """Test distance between units with only horizontal separation."""
        unit_a = MockUnit(52.0, 8.0, 1000.0)
        unit_b = MockUnit(52.001, 8.001, 1000.0)  # Same altitude

        distance = units_distance_km(unit_a, unit_b)

        assert distance > 0.0
        # Should be small distance (roughly 0.1-0.15 km)
        assert distance < 1.0

    def test_units_distance_vertical_only(self):
        """Test distance between units with only vertical separation."""
        unit_a = MockUnit(52.0, 8.0, 1000.0)
        unit_b = MockUnit(52.0, 8.0, 2000.0)  # 1km altitude difference

        distance = units_distance_km(unit_a, unit_b)

        # Should be approximately 1 km (vertical distance)
        assert math.isclose(distance, 1.0, rel_tol=0.01)

    def test_units_distance_3d_separation(self):
        """Test distance between units with 3D separation."""
        unit_a = MockUnit(52.0, 8.0, 1000.0)
        unit_b = MockUnit(52.01, 8.01, 2000.0)  # Both horizontal and vertical separation

        distance = units_distance_km(unit_a, unit_b)

        assert distance > 0.0
        # Should be greater than just vertical or horizontal distance alone
        assert distance > 1.0
        assert distance < 10.0  # Reasonable upper bound

    def test_units_distance_large_separation(self):
        """Test distance between units with large separation."""
        unit_a = MockUnit(52.0, 8.0, 1000.0)
        unit_b = MockUnit(53.0, 9.0, 5000.0)  # Large separation

        distance = units_distance_km(unit_a, unit_b)

        assert distance > 100.0  # Should be substantial distance
        assert distance < 200.0  # Reasonable upper bound for this case

    def test_units_distance_symmetry(self):
        """Test that distance is symmetric (A to B = B to A)."""
        unit_a = MockUnit(52.0, 8.0, 1000.0)
        unit_b = MockUnit(52.1, 8.1, 2000.0)

        distance_ab = units_distance_km(unit_a, unit_b)
        distance_ba = units_distance_km(unit_b, unit_a)

        assert math.isclose(distance_ab, distance_ba, rel_tol=1e-10)

    def test_units_distance_negative_coordinates(self):
        """Test distance with negative coordinates."""
        unit_a = MockUnit(-45.0, -120.0, 500.0)
        unit_b = MockUnit(-45.1, -120.1, 1500.0)

        distance = units_distance_km(unit_a, unit_b)

        assert distance > 0.0
        assert math.isfinite(distance)

    def test_units_distance_zero_altitude(self):
        """Test distance with zero altitude."""
        unit_a = MockUnit(52.0, 8.0, 0.0)
        unit_b = MockUnit(52.01, 8.01, 0.0)

        distance = units_distance_km(unit_a, unit_b)

        assert distance > 0.0
        assert math.isfinite(distance)

    def test_units_distance_very_close(self):
        """Test distance between very close units."""
        unit_a = MockUnit(52.000000, 8.000000, 1000.0)
        unit_b = MockUnit(52.000001, 8.000001, 1000.1)

        distance = units_distance_km(unit_a, unit_b)

        assert distance >= 0.0
        assert distance < 0.001  # Very small distance

    def test_units_distance_antipodal(self):
        """Test distance between antipodal points."""
        unit_a = MockUnit(0.0, 0.0, 0.0)
        unit_b = MockUnit(0.0, 180.0, 0.0)  # Opposite side of Earth

        distance = units_distance_km(unit_a, unit_b)

        # Should be approximately half Earth's circumference
        assert distance > 15000.0  # Rough lower bound
        assert distance < 25000.0  # Rough upper bound


class TestUnitsBearing:
    """Test units_bearing function."""

    def test_units_bearing_same_position(self):
        """Test bearing between units at same position."""
        unit_a = MockUnit(52.0, 8.0, 1000.0)
        unit_b = MockUnit(52.0, 8.0, 1000.0)

        bearing = units_bearing(unit_a, unit_b)

        # Bearing at same position is undefined, but function should return something
        assert 0.0 <= bearing <= 360.0

    def test_units_bearing_north(self):
        """Test bearing for northward direction."""
        unit_a = MockUnit(52.0, 8.0, 1000.0)
        unit_b = MockUnit(53.0, 8.0, 1000.0)  # 1 degree north

        bearing = units_bearing(unit_a, unit_b)

        # Should be approximately 0 degrees (north)
        assert bearing < 10.0 or bearing > 350.0

    def test_units_bearing_south(self):
        """Test bearing for southward direction."""
        unit_a = MockUnit(53.0, 8.0, 1000.0)
        unit_b = MockUnit(52.0, 8.0, 1000.0)  # 1 degree south

        bearing = units_bearing(unit_a, unit_b)

        # Should be approximately 180 degrees (south)
        assert 170.0 < bearing < 190.0

    def test_units_bearing_east(self):
        """Test bearing for eastward direction."""
        unit_a = MockUnit(52.0, 8.0, 1000.0)
        unit_b = MockUnit(52.0, 9.0, 1000.0)  # 1 degree east

        bearing = units_bearing(unit_a, unit_b)

        # Should be approximately 90 degrees (east)
        assert 80.0 < bearing < 100.0

    def test_units_bearing_west(self):
        """Test bearing for westward direction."""
        unit_a = MockUnit(52.0, 9.0, 1000.0)
        unit_b = MockUnit(52.0, 8.0, 1000.0)  # 1 degree west

        bearing = units_bearing(unit_a, unit_b)

        # Should be approximately 270 degrees (west)
        assert 260.0 < bearing < 280.0

    def test_units_bearing_northeast(self):
        """Test bearing for northeast direction."""
        unit_a = MockUnit(52.0, 8.0, 1000.0)
        unit_b = MockUnit(52.1, 8.1, 1000.0)  # Northeast

        bearing = units_bearing(unit_a, unit_b)

        # Should be approximately 45 degrees (northeast)
        assert 30.0 < bearing < 60.0

    def test_units_bearing_southwest(self):
        """Test bearing for southwest direction."""
        unit_a = MockUnit(52.1, 8.1, 1000.0)
        unit_b = MockUnit(52.0, 8.0, 1000.0)  # Southwest

        bearing = units_bearing(unit_a, unit_b)

        # Should be approximately 225 degrees (southwest)
        assert 210.0 < bearing < 240.0

    def test_units_bearing_range(self):
        """Test that bearing is always in valid range [0, 360)."""
        test_positions = [
            (MockUnit(0.0, 0.0, 0), MockUnit(1.0, 1.0, 0)),
            (MockUnit(-45.0, -120.0, 0), MockUnit(45.0, 120.0, 0)),
            (MockUnit(89.0, 179.0, 0), MockUnit(-89.0, -179.0, 0)),
        ]

        for unit_a, unit_b in test_positions:
            bearing = units_bearing(unit_a, unit_b)
            assert 0.0 <= bearing <= 360.0

    def test_units_bearing_altitude_independence(self):
        """Test that bearing is independent of altitude."""
        unit_a1 = MockUnit(52.0, 8.0, 1000.0)
        unit_b1 = MockUnit(52.1, 8.1, 1000.0)

        unit_a2 = MockUnit(52.0, 8.0, 5000.0)
        unit_b2 = MockUnit(52.1, 8.1, 3000.0)

        bearing1 = units_bearing(unit_a1, unit_b1)
        bearing2 = units_bearing(unit_a2, unit_b2)

        # Bearings should be very close (altitude doesn't affect bearing)
        assert math.isclose(bearing1, bearing2, rel_tol=1e-6)

    def test_units_bearing_precision(self):
        """Test bearing precision with small differences."""
        unit_a = MockUnit(52.000000, 8.000000, 1000.0)
        unit_b = MockUnit(52.000001, 8.000001, 1000.0)

        bearing = units_bearing(unit_a, unit_b)

        # Should still produce valid bearing
        assert 0.0 <= bearing <= 360.0
        assert math.isfinite(bearing)


class TestClamp:
    """Test clamp utility function."""

    def test_clamp_within_range(self):
        """Test clamp when value is within range."""
        result = clamp(5, 0, 10)
        assert result == 5

        result_float = clamp(5.5, 0.0, 10.0)
        assert result_float == 5.5

    def test_clamp_below_range(self):
        """Test clamp when value is below range."""
        result = clamp(-5, 0, 10)
        assert result == 0

        result_float = clamp(-2.5, 0.0, 10.0)
        assert result_float == 0.0

    def test_clamp_above_range(self):
        """Test clamp when value is above range."""
        result = clamp(15, 0, 10)
        assert result == 10

        result_float = clamp(12.5, 0.0, 10.0)
        assert result_float == 10.0

    def test_clamp_at_boundaries(self):
        """Test clamp when value is exactly at boundaries."""
        result_low = clamp(0, 0, 10)
        assert result_low == 0

        result_high = clamp(10, 0, 10)
        assert result_high == 10

    def test_clamp_inverted_range(self):
        """Test clamp when lo > hi (should swap)."""
        result = clamp(5, 10, 0)  # lo=10, hi=0 (inverted)
        assert result == 5  # Should be clamped to [0, 10]

        result_below = clamp(-5, 10, 0)
        assert result_below == 0

        result_above = clamp(15, 10, 0)
        assert result_above == 10

    def test_clamp_equal_bounds(self):
        """Test clamp when lo == hi."""
        result = clamp(5, 7, 7)
        assert result == 7

        result_equal = clamp(7, 7, 7)
        assert result_equal == 7

    def test_clamp_negative_values(self):
        """Test clamp with negative values."""
        result = clamp(-5, -10, -1)
        assert result == -5

        result_below = clamp(-15, -10, -1)
        assert result_below == -10

        result_above = clamp(0, -10, -1)
        assert result_above == -1

    def test_clamp_float_precision(self):
        """Test clamp with floating point precision."""
        result = clamp(3.14159, 0.0, 5.0)
        assert result == 3.14159

        result_precise = clamp(1.000000001, 1.0, 2.0)
        assert result_precise == 1.000000001

    def test_clamp_type_preservation(self):
        """Test that clamp preserves input type."""
        # Integer
        int_result = clamp(5, 0, 10)
        assert isinstance(int_result, int)

        # Float
        float_result = clamp(5.0, 0.0, 10.0)
        assert isinstance(float_result, float)

    def test_clamp_edge_cases(self):
        """Test clamp with edge case values."""
        # Very large numbers
        result_large = clamp(1e10, 0, 1e9)
        assert result_large == 1e9

        # Very small numbers
        result_small = clamp(1e-10, 1e-9, 1e-8)
        assert result_small == 1e-9

        # Zero boundaries
        result_zero = clamp(0.5, 0, 0)
        assert result_zero == 0


class TestHelpersIntegration:
    """Integration tests for helpers module."""

    def test_position_and_distance_integration(self):
        """Test Position with distance calculation."""
        pos1 = Position(52.0, 8.0, 1000.0)
        pos2 = Position(52.1, 8.1, 2000.0)

        unit_a = MockUnit(pos1.lat, pos1.lon, pos1.alt)
        unit_b = MockUnit(pos2.lat, pos2.lon, pos2.alt)

        distance = units_distance_km(unit_a, unit_b)
        bearing = units_bearing(unit_a, unit_b)

        assert distance > 0.0
        assert 0.0 <= bearing <= 360.0

    def test_position_copy_with_calculations(self):
        """Test Position copy with distance/bearing calculations."""
        original_pos = Position(52.0, 8.0, 1000.0)
        copied_pos = original_pos.copy()

        unit_original = MockUnit(original_pos.lat, original_pos.lon, original_pos.alt)
        unit_copied = MockUnit(copied_pos.lat, copied_pos.lon, copied_pos.alt)

        distance = units_distance_km(unit_original, unit_copied)
        assert distance == 0.0  # Same position

    def test_clamp_with_position_coordinates(self):
        """Test clamp utility with position coordinates."""
        # Clamp latitude to valid range
        lat = clamp(95.0, -90.0, 90.0)  # Invalid latitude
        lon = clamp(185.0, -180.0, 180.0)  # Invalid longitude
        alt = clamp(-500.0, 0.0, 50000.0)  # Below sea level

        pos = Position(lat, lon, alt)

        assert pos.lat == 90.0  # Clamped to max
        assert pos.lon == 180.0  # Clamped to max
        assert pos.alt == 0.0  # Clamped to min
