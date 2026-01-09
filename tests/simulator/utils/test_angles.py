#!/usr/bin/env python3
"""
Comprehensive tests for simulator.utils.angles module.
Tests angle utility functions including normalization, conversion,
and geometric calculations.
"""

import pytest
import math
from simulator.utils.angles import (
    deg2rad, rad2deg, normalize_angle, sum_angles, signed_yaw_deg_diff,
    clamp_pitch_deg, yaw_geo_to_math, yaw_math_to_geo
)


def is_angle_close(a, b, tol=1e-9):
    """Test numerically equivalent angles on circle."""
    return abs(((a - b + 180) % 360) - 180) < tol or abs(((a - b) % 360)) < tol


class TestDegreeRadianConversion:
    """Test degree/radian conversion functions."""
    
    def test_deg2rad_basic(self):
        """Test basic degree to radian conversion."""
        assert math.isclose(deg2rad(0), 0.0, abs_tol=1e-10)
        assert math.isclose(deg2rad(90), math.pi/2, rel_tol=1e-10)
        assert math.isclose(deg2rad(180), math.pi, rel_tol=1e-10)
        assert math.isclose(deg2rad(270), 3*math.pi/2, rel_tol=1e-10)
        assert math.isclose(deg2rad(360), 2*math.pi, rel_tol=1e-10)
    
    def test_deg2rad_negative(self):
        """Test degree to radian conversion with negative values."""
        assert math.isclose(deg2rad(-90), -math.pi/2, rel_tol=1e-10)
        assert math.isclose(deg2rad(-180), -math.pi, rel_tol=1e-10)
        assert math.isclose(deg2rad(-360), -2*math.pi, rel_tol=1e-10)
    
    def test_deg2rad_fractional(self):
        """Test degree to radian conversion with fractional degrees."""
        assert math.isclose(deg2rad(45), math.pi/4, rel_tol=1e-10)
        assert math.isclose(deg2rad(30), math.pi/6, rel_tol=1e-10)
        assert math.isclose(deg2rad(60), math.pi/3, rel_tol=1e-10)
    
    def test_deg2rad_large_values(self):
        """Test degree to radian conversion with large values."""
        assert math.isclose(deg2rad(720), 4*math.pi, rel_tol=1e-10)
        assert math.isclose(deg2rad(1080), 6*math.pi, rel_tol=1e-10)
    
    def test_rad2deg_basic(self):
        """Test basic radian to degree conversion."""
        assert math.isclose(rad2deg(0), 0.0, abs_tol=1e-10)
        assert math.isclose(rad2deg(math.pi/2), 90.0, rel_tol=1e-10)
        assert math.isclose(rad2deg(math.pi), 180.0, rel_tol=1e-10)
        assert math.isclose(rad2deg(3*math.pi/2), 270.0, rel_tol=1e-10)
        assert math.isclose(rad2deg(2*math.pi), 360.0, rel_tol=1e-10)
    
    def test_rad2deg_negative(self):
        """Test radian to degree conversion with negative values."""
        assert math.isclose(rad2deg(-math.pi/2), -90.0, rel_tol=1e-10)
        assert math.isclose(rad2deg(-math.pi), -180.0, rel_tol=1e-10)
        assert math.isclose(rad2deg(-2*math.pi), -360.0, rel_tol=1e-10)
    
    def test_rad2deg_fractional(self):
        """Test radian to degree conversion with fractional radians."""
        assert math.isclose(rad2deg(math.pi/4), 45.0, rel_tol=1e-10)
        assert math.isclose(rad2deg(math.pi/6), 30.0, rel_tol=1e-10)
        assert math.isclose(rad2deg(math.pi/3), 60.0, rel_tol=1e-10)
    
    def test_conversion_roundtrip(self):
        """Test roundtrip conversion deg->rad->deg."""
        test_angles = [0, 30, 45, 60, 90, 120, 180, 270, 360, -90, -180]
        
        for angle in test_angles:
            converted = rad2deg(deg2rad(angle))
            assert math.isclose(converted, angle, rel_tol=1e-10)
    
    def test_conversion_precision(self):
        """Test conversion precision with arbitrary values."""
        test_values = [123.456, -67.89, 0.001, 359.999]
        
        for deg_val in test_values:
            rad_val = deg2rad(deg_val)
            back_to_deg = rad2deg(rad_val)
            assert math.isclose(back_to_deg, deg_val, rel_tol=1e-10)


class TestNormalizeAngle:
    """Test angle normalization function."""
    
    def test_normalize_angle_basic(self):
        """Test basic angle normalization."""
        assert is_angle_close(normalize_angle(0), 0)
        assert is_angle_close(normalize_angle(90), 90)
        assert is_angle_close(normalize_angle(180), 180)
        assert is_angle_close(normalize_angle(270), 270)
    
    def test_normalize_angle_boundary(self):
        """Test angle normalization at boundaries."""
        # 360 should normalize to 0
        result_360 = normalize_angle(360)
        assert is_angle_close(result_360, 0) or is_angle_close(result_360, 360)
    
    def test_normalize_angle_negative(self):
        """Test angle normalization with negative values."""
        assert is_angle_close(normalize_angle(-90), 270)
        assert is_angle_close(normalize_angle(-180), 180)
        assert is_angle_close(normalize_angle(-270), 90)
    
    def test_normalize_angle_large_positive(self):
        """Test angle normalization with large positive values."""
        assert is_angle_close(normalize_angle(370), 10)
        assert is_angle_close(normalize_angle(450), 90)
        assert is_angle_close(normalize_angle(720), 0)
    
    def test_normalize_angle_large_negative(self):
        """Test angle normalization with large negative values."""
        assert is_angle_close(normalize_angle(-370), 350)
        assert is_angle_close(normalize_angle(-450), 270)
        assert is_angle_close(normalize_angle(-720), 0)
    
    def test_normalize_angle_fractional(self):
        """Test angle normalization with fractional values."""
        assert is_angle_close(normalize_angle(45.5), 45.5)
        assert is_angle_close(normalize_angle(360.25), 0.25)
        assert is_angle_close(normalize_angle(-45.75), 314.25)
    
    def test_normalize_angle_range(self):
        """Test that normalized angle is always in [0, 360)."""
        test_angles = [-1000, -500, -180, -90, 0, 90, 180, 270, 360, 450, 720, 1000]
        
        for angle in test_angles:
            normalized = normalize_angle(angle)
            # Handle the fact that 360 may be returned instead of 0 due to floating point precision
            if abs(normalized - 360.0) < 1e-10:
                normalized = 0.0
            assert 0 <= normalized < 360
    
    def test_normalize_angle_precision(self):
        """Test angle normalization precision."""
        # Very small angles
        assert is_angle_close(normalize_angle(0.001), 0.001)
        assert is_angle_close(normalize_angle(-0.001), 359.999)
        
        # Near boundary angles
        assert is_angle_close(normalize_angle(359.999), 359.999)
        assert is_angle_close(normalize_angle(360.001), 0.001)
    
    def test_normalize_angle_arbitrary_values(self):
        """Test angle normalization with arbitrary values."""
        test_cases = [
            (123.456, 123.456),
            (123.456 + 360, 123.456),
            (123.456 - 360, 123.456),
            (123.456 + 720, 123.456),
        ]
        
        for input_angle, expected in test_cases:
            result = normalize_angle(input_angle)
            assert is_angle_close(result, expected)


class TestSumAngles:
    """Test angle sum function."""
    
    def test_sum_angles_basic(self):
        """Test basic angle addition."""
        assert is_angle_close(sum_angles(0, 0), 0)
        assert is_angle_close(sum_angles(90, 90), 180)
        assert is_angle_close(sum_angles(180, 180), 0)  # 360 -> 0
        assert is_angle_close(sum_angles(45, 45), 90)
    
    def test_sum_angles_wrap_around(self):
        """Test angle sum with wrap-around."""
        assert is_angle_close(sum_angles(350, 20), 10)
        assert is_angle_close(sum_angles(270, 180), 90)  # 450 -> 90
        assert is_angle_close(sum_angles(200, 200), 40)  # 400 -> 40
    
    def test_sum_angles_negative(self):
        """Test angle sum with negative values."""
        assert is_angle_close(sum_angles(90, -270), 180)
        assert is_angle_close(sum_angles(-45, 45), 0)
        assert is_angle_close(sum_angles(-180, -180), 0)  # -360 -> 0
    
    def test_sum_angles_large_values(self):
        """Test angle sum with large values."""
        assert is_angle_close(sum_angles(720, 360), 0)  # 1080 -> 0
        assert is_angle_close(sum_angles(450, 270), 0)  # 720 -> 0
        assert is_angle_close(sum_angles(1000, 1000), 200)  # 2000 -> 200
    
    def test_sum_angles_fractional(self):
        """Test angle sum with fractional values."""
        assert is_angle_close(sum_angles(45.5, 44.5), 90.0)
        assert is_angle_close(sum_angles(180.25, 179.75), 0.0)  # 360 -> 0
        assert is_angle_close(sum_angles(123.456, 76.544), 200.0)
    
    def test_sum_angles_commutativity(self):
        """Test that angle sum is commutative."""
        test_pairs = [
            (45, 135),
            (90, 270),
            (123.456, 67.89),
            (-90, 450),
            (350, 20)
        ]
        
        for a, b in test_pairs:
            sum_ab = sum_angles(a, b)
            sum_ba = sum_angles(b, a)
            assert is_angle_close(sum_ab, sum_ba)
    
    def test_sum_angles_zero_identity(self):
        """Test that adding zero doesn't change angle."""
        test_angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        
        for angle in test_angles:
            result = sum_angles(angle, 0)
            assert is_angle_close(result, normalize_angle(angle))


class TestSignedYawDegDiff:
    """Test signed yaw difference function."""
    
    def test_signed_yaw_deg_diff_basic(self):
        """Test basic signed yaw difference."""
        assert math.isclose(signed_yaw_deg_diff(0, 90), 90)
        assert math.isclose(signed_yaw_deg_diff(90, 0), -90)
        assert math.isclose(signed_yaw_deg_diff(0, 180), 180)
        assert math.isclose(signed_yaw_deg_diff(180, 0), -180)
    
    def test_signed_yaw_deg_diff_same_angle(self):
        """Test signed yaw difference with same angles."""
        test_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        for angle in test_angles:
            diff = signed_yaw_deg_diff(angle, angle)
            assert math.isclose(diff, 0.0, abs_tol=1e-10)
    
    def test_signed_yaw_deg_diff_wrap_around(self):
        """Test signed yaw difference with wrap-around."""
        # Shortest path from 350 to 10 is +20
        assert math.isclose(signed_yaw_deg_diff(350, 10), 20)
        # Shortest path from 10 to 350 is -20
        assert math.isclose(signed_yaw_deg_diff(10, 350), -20)
    
    def test_signed_yaw_deg_diff_180_degree(self):
        """Test signed yaw difference at 180-degree boundary."""
        # At 180 degrees, could go either way - function should be consistent
        diff_1 = signed_yaw_deg_diff(0, 180)
        diff_2 = signed_yaw_deg_diff(180, 0)
        
        assert abs(abs(diff_1) - 180) < 1e-10
        assert abs(abs(diff_2) - 180) < 1e-10
        assert math.isclose(diff_1, -diff_2, abs_tol=1e-10)
    
    def test_signed_yaw_deg_diff_small_angles(self):
        """Test signed yaw difference with small angles."""
        assert math.isclose(signed_yaw_deg_diff(1, 2), 1)
        assert math.isclose(signed_yaw_deg_diff(2, 1), -1)
        assert math.isclose(signed_yaw_deg_diff(0.1, 0.2), 0.1)
    
    def test_signed_yaw_deg_diff_range(self):
        """Test that signed yaw difference is in range (-180, 180]."""
        test_pairs = [
            (0, 90), (90, 180), (180, 270), (270, 0),
            (45, 225), (315, 135), (10, 350), (350, 10)
        ]
        
        for actual, desired in test_pairs:
            diff = signed_yaw_deg_diff(actual, desired)
            assert -180 <= diff <= 180
    
    def test_signed_yaw_deg_diff_compatibility(self):
        """Test compatibility with existing test cases."""
        assert math.isclose(signed_yaw_deg_diff(10, 20), 10)
        assert math.isclose(signed_yaw_deg_diff(350, 10), 20)
        assert math.isclose(signed_yaw_deg_diff(10, 350), -20)
    
    def test_signed_yaw_deg_diff_fractional(self):
        """Test signed yaw difference with fractional degrees."""
        assert math.isclose(signed_yaw_deg_diff(10.5, 20.5), 10.0)
        assert math.isclose(signed_yaw_deg_diff(350.25, 10.75), 20.5)
        assert math.isclose(signed_yaw_deg_diff(123.456, 234.567), 111.111, rel_tol=1e-3)


class TestClampPitchDeg:
    """Test pitch angle clamping function."""
    
    def test_clamp_pitch_deg_within_range(self):
        """Test pitch clamping when within valid range."""
        assert clamp_pitch_deg(0) == 0
        assert clamp_pitch_deg(45) == 45
        assert clamp_pitch_deg(-45) == -45
        assert clamp_pitch_deg(89) == 89
        assert clamp_pitch_deg(-89) == -89
    
    def test_clamp_pitch_deg_at_limits(self):
        """Test pitch clamping at exact limits."""
        assert clamp_pitch_deg(90) == 90
        assert clamp_pitch_deg(-90) == -90
    
    def test_clamp_pitch_deg_above_limit(self):
        """Test pitch clamping above upper limit."""
        assert clamp_pitch_deg(100) == 90
        assert clamp_pitch_deg(120) == 90
        assert clamp_pitch_deg(180) == 90
        assert clamp_pitch_deg(1000) == 90
    
    def test_clamp_pitch_deg_below_limit(self):
        """Test pitch clamping below lower limit."""
        assert clamp_pitch_deg(-100) == -90
        assert clamp_pitch_deg(-120) == -90
        assert clamp_pitch_deg(-180) == -90
        assert clamp_pitch_deg(-1000) == -90
    
    def test_clamp_pitch_deg_fractional(self):
        """Test pitch clamping with fractional degrees."""
        assert clamp_pitch_deg(45.5) == 45.5
        assert clamp_pitch_deg(-67.89) == -67.89
        assert clamp_pitch_deg(90.1) == 90
        assert clamp_pitch_deg(-90.1) == -90
    
    def test_clamp_pitch_deg_precision(self):
        """Test pitch clamping with high precision values."""
        assert clamp_pitch_deg(89.999999) == 89.999999
        assert clamp_pitch_deg(-89.999999) == -89.999999
        assert clamp_pitch_deg(90.000001) == 90
        assert clamp_pitch_deg(-90.000001) == -90


class TestYawConversions:
    """Test yaw conversion functions between geographic and mathematical."""
    
    def test_yaw_geo_to_math_basic(self):
        """Test basic geographic to mathematical yaw conversion."""
        # 0° Geographic (North) = 90° Mathematical
        assert math.isclose(yaw_geo_to_math(0), 90.0)
        # 90° Geographic (East) = 0° Mathematical  
        assert math.isclose(yaw_geo_to_math(90), 0.0)
        # 180° Geographic (South) = -90° Mathematical
        assert math.isclose(yaw_geo_to_math(180), -90.0)
        # 270° Geographic (West) = -180° Mathematical
        assert math.isclose(yaw_geo_to_math(270), -180.0)
    
    def test_yaw_math_to_geo_basic(self):
        """Test basic mathematical to geographic yaw conversion."""
        # 0° Mathematical = 90° Geographic (East)
        assert math.isclose(yaw_math_to_geo(0), 90.0)
        # 90° Mathematical = 0° Geographic (North)
        assert math.isclose(yaw_math_to_geo(90), 0.0)
        # -90° Mathematical = 180° Geographic (South)
        assert math.isclose(yaw_math_to_geo(-90), 180.0)
        # -180° Mathematical = 270° Geographic (West)
        assert math.isclose(yaw_math_to_geo(-180), 270.0)
    
    def test_yaw_conversion_roundtrip(self):
        """Test roundtrip yaw conversions."""
        test_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        for angle in test_angles:
            # Geographic -> Mathematical -> Geographic
            math_angle = yaw_geo_to_math(angle)
            back_to_geo = yaw_math_to_geo(math_angle)
            assert math.isclose(back_to_geo, angle, rel_tol=1e-10)
            
            # Mathematical -> Geographic -> Mathematical
            geo_angle = yaw_math_to_geo(angle)
            back_to_math = yaw_geo_to_math(geo_angle)
            # Check if angles are equivalent (differ by multiple of 360°)
            angle_diff = abs(back_to_math - angle)
            assert angle_diff < 1e-10 or math.isclose(angle_diff, 360.0, rel_tol=1e-10)
    
    def test_yaw_conversion_fractional(self):
        """Test yaw conversions with fractional degrees."""
        # Test specific fractional values
        geo_angle = 45.5
        math_angle = yaw_geo_to_math(geo_angle)
        assert math.isclose(math_angle, 44.5)
        
        # Roundtrip test
        back_to_geo = yaw_math_to_geo(math_angle)
        assert math.isclose(back_to_geo, geo_angle, rel_tol=1e-10)
    
    def test_yaw_conversion_edge_cases(self):
        """Test yaw conversions with edge case values."""
        # Very small angles
        small_geo = 0.001
        small_math = yaw_geo_to_math(small_geo)
        assert math.isclose(small_math, 89.999)
        
        # Very large angles (should still work)
        large_geo = 360.0 + 45.0  # 405°
        large_math = yaw_geo_to_math(large_geo)
        expected_math = yaw_geo_to_math(45.0)  # Should be equivalent to 45°
        assert math.isclose(large_math, expected_math)
    
    def test_yaw_conversion_negative_angles(self):
        """Test yaw conversions with negative angles."""
        neg_geo = -45.0
        neg_math = yaw_geo_to_math(neg_geo)
        # -45° geo should give same mathematical angle as 315° geo
        # Expected mathematical angle is equivalent to 135° (or -225°)
        expected = 135.0
        angle_diff = abs(neg_math - expected)
        assert angle_diff < 1e-10 or math.isclose(angle_diff, 360.0, rel_tol=1e-10)
        
        # Roundtrip
        back_to_geo = yaw_math_to_geo(neg_math)
        # Note: may not be exactly -45 due to angle normalization
        # but should be equivalent angle
        equivalent_geo = (-45.0 + 360.0) % 360  # 315°
        assert math.isclose(back_to_geo, equivalent_geo) or math.isclose(back_to_geo, -45.0)
    
    def test_yaw_conversion_compatibility(self):
        """Test compatibility with existing conversion tests."""
        # Geographic to Mathematical
        assert math.isclose(yaw_geo_to_math(0), 90.0)
        assert math.isclose(yaw_geo_to_math(90), 0.0)
        
        # Mathematical to Geographic  
        assert math.isclose(yaw_math_to_geo(0), 90.0)
        assert math.isclose(yaw_math_to_geo(90), 0.0)
        
        # Roundtrip test
        test_val = 123.4
        assert math.isclose(yaw_math_to_geo(yaw_geo_to_math(test_val)), test_val)


class TestAnglesIntegration:
    """Integration tests for angle utilities."""
    
    def test_navigation_calculations(self):
        """Test angle utilities in navigation context."""
        # Aircraft heading North (0°), wants to turn East (90°)
        current_heading = 0.0
        desired_heading = 90.0
        
        # Calculate turn amount
        turn_amount = signed_yaw_deg_diff(current_heading, desired_heading)
        assert math.isclose(turn_amount, 90.0)
        
        # Apply turn
        new_heading = sum_angles(current_heading, turn_amount)
        assert is_angle_close(new_heading, 90.0)
    
    def test_attitude_limiting(self):
        """Test attitude angle limiting."""
        # Excessive pitch angles should be clamped
        excessive_pitch_up = 120.0
        excessive_pitch_down = -100.0
        
        clamped_up = clamp_pitch_deg(excessive_pitch_up)
        clamped_down = clamp_pitch_deg(excessive_pitch_down)
        
        assert clamped_up == 90.0
        assert clamped_down == -90.0
    
    def test_coordinate_system_conversion(self):
        """Test conversion between coordinate systems."""
        # North in geographic coordinates
        geo_north = 0.0
        math_north = yaw_geo_to_math(geo_north)
        
        # Should be 90° in mathematical coordinates
        assert math.isclose(math_north, 90.0)
        
        # Convert back
        back_to_geo = yaw_math_to_geo(math_north)
        assert math.isclose(back_to_geo, 0.0)
    
    def test_angle_arithmetic_chain(self):
        """Test chain of angle arithmetic operations."""
        initial_angle = 45.0
        
        # Series of operations
        angle1 = sum_angles(initial_angle, 90.0)  # 135°
        angle2 = normalize_angle(angle1 + 180.0)  # 315°
        angle3 = sum_angles(angle2, 90.0)         # 45° (back to start + full circle)
        
        # Should be back close to initial (plus full rotation)
        assert is_angle_close(angle3, initial_angle)
    
    def test_precision_preservation(self):
        """Test that precision is preserved through operations."""
        precise_angle = 123.456789
        
        # Convert to radians and back
        rad_angle = deg2rad(precise_angle)
        back_to_deg = rad2deg(rad_angle)
        
        assert math.isclose(back_to_deg, precise_angle, rel_tol=1e-10)
        
        # Normalize and check precision
        normalized = normalize_angle(precise_angle)
        assert math.isclose(normalized, precise_angle, rel_tol=1e-10)


class TestAnglesCompatibility:
    """Test compatibility with existing angle tests."""
    
    def test_normalize_angle_compatibility(self):
        """Test compatibility with existing normalize_angle tests."""
        assert is_angle_close(normalize_angle(0), 0)
        assert is_angle_close(normalize_angle(360), 0) or is_angle_close(normalize_angle(360), 360)
        assert is_angle_close(normalize_angle(-90), 270)
        assert is_angle_close(normalize_angle(370), 10)
        assert 0 <= normalize_angle(123.456) < 360
    
    def test_sum_angles_compatibility(self):
        """Test compatibility with existing sum_angles tests."""
        assert is_angle_close(sum_angles(350, 20), 10)
        assert is_angle_close(sum_angles(180, 180), 0)
        assert is_angle_close(sum_angles(90, -270), 180)
    
    def test_signed_yaw_deg_diff_compatibility(self):
        """Test compatibility with existing signed_yaw_deg_diff tests."""
        assert math.isclose(signed_yaw_deg_diff(10, 20), 10)
        assert math.isclose(signed_yaw_deg_diff(350, 10), 20)
        assert math.isclose(signed_yaw_deg_diff(10, 350), -20)
    
    def test_clamp_pitch_deg_compatibility(self):
        """Test compatibility with existing clamp_pitch_deg tests."""
        assert clamp_pitch_deg(100) == 90
        assert clamp_pitch_deg(-100) == -90
        assert clamp_pitch_deg(0) == 0
    
    def test_yaw_conversions_compatibility(self):
        """Test compatibility with existing yaw conversion tests."""
        # Geographic to Mathematical
        assert math.isclose(yaw_geo_to_math(0), 90.0)
        assert math.isclose(yaw_geo_to_math(90), 0.0)
        
        # Mathematical to Geographic
        assert math.isclose(yaw_math_to_geo(0), 90.0)
        assert math.isclose(yaw_math_to_geo(90), 0.0)
        
        # Roundtrip
        val = 123.4
        assert math.isclose(yaw_math_to_geo(yaw_geo_to_math(val)), val)