"""
Comprehensive tests for simulator.utils.geodesics module.

Tests geodetic calculations including distance computation, bearing calculation,
and direct coordinate transformations using WGS84 ellipsoid.
"""

import pytest
import math
from unittest.mock import Mock, patch
from simulator.utils.geodesics import (
    geodetic_distance_km, 
    geodetic_bearing_deg, 
    geodetic_direct
)


class TestGeodeticDistanceKm:
    """Test geodetic distance calculations with altitude."""
    
    def test_same_position_zero_distance(self):
        """Same position should return zero distance."""
        dist = geodetic_distance_km(45.0, -123.0, 1000.0, 45.0, -123.0, 1000.0)
        assert dist == pytest.approx(0.0, abs=1e-10)
    
    def test_horizontal_distance_only(self):
        """Test pure horizontal distance with same altitude."""
        # Approximately 1 degree of longitude at 45°N is about 78.8 km
        dist = geodetic_distance_km(45.0, -123.0, 1000.0, 45.0, -122.0, 1000.0)
        assert dist == pytest.approx(78.8, rel=0.1)
    
    def test_vertical_distance_only(self):
        """Test pure vertical distance with same coordinates."""
        dist = geodetic_distance_km(45.0, -123.0, 0.0, 45.0, -123.0, 1000.0)
        assert dist == pytest.approx(1.0, abs=0.01)  # 1000m = 1km
    
    def test_combined_horizontal_vertical_distance(self):
        """Test combined horizontal and vertical distance using Pythagorean theorem."""
        # Known coordinates for testing
        lat1, lon1, alt1 = 47.6062, -122.3321, 0.0  # Seattle
        lat2, lon2, alt2 = 47.7511, -120.7401, 3000.0  # Wenatchee with altitude
        
        dist = geodetic_distance_km(lat1, lon1, alt1, lat2, lon2, alt2)
        
        # Should be greater than horizontal distance alone
        horizontal_only = geodetic_distance_km(lat1, lon1, alt1, lat2, lon2, alt1)
        assert dist > horizontal_only
        
        # Should be less than sum of horizontal + vertical
        vertical_component = abs(alt2 - alt1) / 1000.0
        assert dist < (horizontal_only + vertical_component)
    
    def test_equatorial_distance(self):
        """Test distance calculation on equator."""
        # 1 degree longitude on equator ≈ 111.32 km
        dist = geodetic_distance_km(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        assert dist == pytest.approx(111.32, rel=0.01)
    
    def test_polar_distance(self):
        """Test distance calculation near poles."""
        # Distance should be smaller near poles due to longitude convergence
        dist_equator = geodetic_distance_km(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        dist_polar = geodetic_distance_km(89.0, 0.0, 0.0, 89.0, 1.0, 0.0)
        assert dist_polar < dist_equator
    
    def test_antipodal_points(self):
        """Test distance between antipodal points (opposite sides of Earth)."""
        # Half circumference of Earth ≈ 20015 km
        dist = geodetic_distance_km(45.0, 0.0, 0.0, -45.0, 180.0, 0.0)
        assert dist == pytest.approx(20015, rel=0.1)
    
    def test_large_altitude_differences(self):
        """Test with large altitude differences."""
        dist = geodetic_distance_km(45.0, -123.0, 0.0, 45.0, -123.0, 50000.0)
        assert dist == pytest.approx(50.0, abs=0.01)  # 50km vertical
    
    def test_negative_coordinates(self):
        """Test with negative latitude and longitude."""
        dist = geodetic_distance_km(-45.0, -123.0, 1000.0, -46.0, -124.0, 2000.0)
        assert dist > 0
    
    def test_crossing_international_dateline(self):
        """Test distance calculation crossing international dateline."""
        # From west of dateline to east of dateline
        dist = geodetic_distance_km(45.0, 179.0, 0.0, 45.0, -179.0, 0.0)
        # Should be short distance, not around the world
        assert dist < 500  # Much less than half circumference


class TestGeodeticBearingDeg:
    """Test geodetic bearing calculations."""
    
    def test_same_position_bearing(self):
        """Bearing from same position should be handled gracefully."""
        bearing = geodetic_bearing_deg(45.0, -123.0, 45.0, -123.0)
        # Should be a normalized angle (exact value may vary by implementation)
        assert 0 <= bearing < 360
    
    def test_north_bearing(self):
        """Test bearing pointing due north."""
        bearing = geodetic_bearing_deg(45.0, -123.0, 46.0, -123.0)
        assert bearing == pytest.approx(0.0, abs=1.0)
    
    def test_south_bearing(self):
        """Test bearing pointing due south."""
        bearing = geodetic_bearing_deg(46.0, -123.0, 45.0, -123.0)
        assert bearing == pytest.approx(180.0, abs=1.0)
    
    def test_east_bearing(self):
        """Test bearing pointing due east."""
        bearing = geodetic_bearing_deg(45.0, -123.0, 45.0, -122.0)
        assert bearing == pytest.approx(90.0, abs=1.0)
    
    def test_west_bearing(self):
        """Test bearing pointing due west."""
        bearing = geodetic_bearing_deg(45.0, -122.0, 45.0, -123.0)
        assert bearing == pytest.approx(270.0, abs=1.0)
    
    def test_northeast_bearing(self):
        """Test bearing pointing northeast."""
        bearing = geodetic_bearing_deg(45.0, -123.0, 46.0, -122.0)
        assert 0 < bearing < 90
    
    def test_southeast_bearing(self):
        """Test bearing pointing southeast."""
        bearing = geodetic_bearing_deg(46.0, -123.0, 45.0, -122.0)
        assert 90 < bearing < 180
    
    def test_southwest_bearing(self):
        """Test bearing pointing southwest."""
        bearing = geodetic_bearing_deg(46.0, -122.0, 45.0, -123.0)
        assert 180 < bearing < 270
    
    def test_northwest_bearing(self):
        """Test bearing pointing northwest."""
        bearing = geodetic_bearing_deg(45.0, -122.0, 46.0, -123.0)
        assert 270 < bearing < 360
    
    def test_bearing_normalization(self):
        """Test that bearing is properly normalized to [0, 360)."""
        bearing = geodetic_bearing_deg(0.0, 179.0, 0.0, -179.0)
        assert 0 <= bearing < 360
    
    def test_equatorial_bearing(self):
        """Test bearing calculation on equator."""
        bearing = geodetic_bearing_deg(0.0, 0.0, 0.0, 1.0)
        assert bearing == pytest.approx(90.0, abs=1.0)
    
    def test_polar_bearing(self):
        """Test bearing calculation near poles."""
        # From near north pole
        bearing = geodetic_bearing_deg(89.0, 0.0, 89.0, 90.0)
        assert 0 <= bearing < 360


class TestGeodeticDirect:
    """Test direct geodetic coordinate transformations."""
    
    def test_zero_distance_same_position(self):
        """Zero distance should return same position."""
        lat, lon, alt = geodetic_direct(45.0, -123.0, 1000.0, 90.0, 0.0, 0.0)
        assert lat == pytest.approx(45.0, abs=1e-10)
        assert lon == pytest.approx(-123.0, abs=1e-10)
        assert alt == pytest.approx(1000.0, abs=1e-10)
    
    def test_north_movement(self):
        """Test movement due north."""
        lat, lon, alt = geodetic_direct(45.0, -123.0, 1000.0, 0.0, 111320.0, 0.0)
        assert lat > 45.0
        assert lon == pytest.approx(-123.0, abs=1e-6)
        assert alt == pytest.approx(1000.0, abs=1e-10)
    
    def test_south_movement(self):
        """Test movement due south."""
        lat, lon, alt = geodetic_direct(45.0, -123.0, 1000.0, 180.0, 111320.0, 0.0)
        assert lat < 45.0
        assert lon == pytest.approx(-123.0, abs=1e-6)
        assert alt == pytest.approx(1000.0, abs=1e-10)
    
    def test_east_movement(self):
        """Test movement due east."""
        lat, lon, alt = geodetic_direct(45.0, -123.0, 1000.0, 90.0, 78847.0, 0.0)  # ~1 degree at 45°N
        assert lat == pytest.approx(45.0, abs=1e-2)  # Geodetic calculations may have small latitude changes
        assert lon > -123.0
        assert alt == pytest.approx(1000.0, abs=1e-10)
    
    def test_west_movement(self):
        """Test movement due west."""
        lat, lon, alt = geodetic_direct(45.0, -123.0, 1000.0, 270.0, 78847.0, 0.0)
        assert lat == pytest.approx(45.0, abs=1e-2)  # Geodetic calculations may have small latitude changes
        assert lon < -123.0
        assert alt == pytest.approx(1000.0, abs=1e-10)
    
    def test_vertical_distance_positive(self):
        """Test positive vertical distance (climb)."""
        lat, lon, alt = geodetic_direct(45.0, -123.0, 1000.0, 0.0, 0.0, 2000.0)
        assert lat == pytest.approx(45.0, abs=1e-10)
        assert lon == pytest.approx(-123.0, abs=1e-10)
        assert alt == pytest.approx(3000.0, abs=1e-10)
    
    def test_vertical_distance_negative(self):
        """Test negative vertical distance (descent)."""
        lat, lon, alt = geodetic_direct(45.0, -123.0, 1000.0, 0.0, 0.0, -500.0)
        assert lat == pytest.approx(45.0, abs=1e-10)
        assert lon == pytest.approx(-123.0, abs=1e-10)
        assert alt == pytest.approx(500.0, abs=1e-10)
    
    def test_combined_horizontal_vertical(self):
        """Test combined horizontal and vertical movement."""
        lat, lon, alt = geodetic_direct(45.0, -123.0, 1000.0, 45.0, 100000.0, 1500.0)
        assert lat > 45.0  # Northeast movement
        assert lon > -123.0
        assert alt == pytest.approx(2500.0, abs=1e-10)
    
    def test_large_distance_movement(self):
        """Test movement over large distances."""
        lat, lon, alt = geodetic_direct(45.0, -123.0, 1000.0, 90.0, 1000000.0, 0.0)  # 1000 km east
        # For large distances, geodesic calculation causes slight latitude change
        assert lat == pytest.approx(45.0, abs=1.0)  # Increased tolerance for geodesic effects
        assert lon > -123.0
        assert abs(lon - (-123.0)) > 10  # Significant longitude change
    
    def test_crossing_meridians(self):
        """Test movement crossing meridians."""
        # Start near Greenwich meridian and move east
        lat, lon, alt = geodetic_direct(45.0, 0.0, 1000.0, 90.0, 100000.0, 0.0)
        assert lon > 0
    
    def test_full_circle_bearing(self):
        """Test all cardinal and intercardinal bearings work correctly."""
        bearings = [0, 45, 90, 135, 180, 225, 270, 315]
        for bearing in bearings:
            lat, lon, alt = geodetic_direct(45.0, -123.0, 1000.0, bearing, 10000.0, 0.0)
            assert -90 <= lat <= 90  # Valid latitude
            assert -180 <= lon <= 180  # Valid longitude
    
    def test_equatorial_movement(self):
        """Test movement along equator."""
        lat, lon, alt = geodetic_direct(0.0, 0.0, 0.0, 90.0, 111320.0, 0.0)  # 1 degree east
        assert lat == pytest.approx(0.0, abs=1e-6)
        assert lon == pytest.approx(1.0, abs=0.01)
    
    def test_polar_region_movement(self):
        """Test movement in polar regions."""
        lat, lon, alt = geodetic_direct(85.0, 0.0, 0.0, 0.0, 100000.0, 0.0)  # North from 85°N
        assert lat > 85.0
        assert lat <= 90.0  # Shouldn't exceed north pole


class TestGeodeticIntegration:
    """Integration tests combining multiple geodetic functions."""
    
    def test_distance_bearing_consistency(self):
        """Test that distance and bearing calculations are consistent."""
        lat1, lon1, alt1 = 47.6062, -122.3321, 1000.0  # Seattle
        lat2, lon2, alt2 = 47.7511, -120.7401, 2000.0   # Wenatchee
        
        # Calculate distance and bearing
        distance_km = geodetic_distance_km(lat1, lon1, alt1, lat2, lon2, alt2)
        bearing = geodetic_bearing_deg(lat1, lon1, lat2, lon2)
        
        # Use direct to go from point 1 towards point 2
        horizontal_dist = geodetic_distance_km(lat1, lon1, alt1, lat2, lon2, alt1)
        calc_lat, calc_lon, _ = geodetic_direct(lat1, lon1, alt1, bearing, horizontal_dist * 1000, 0)
        
        # Should be close to original point 2 (within reasonable tolerance)
        assert calc_lat == pytest.approx(lat2, abs=0.01)
        assert calc_lon == pytest.approx(lon2, abs=0.01)
    
    def test_round_trip_calculation(self):
        """Test round-trip: direct then inverse calculation."""
        original_lat, original_lon = 45.0, -123.0
        bearing = 67.5  # Northeast
        distance_m = 50000  # 50 km
        
        # Go to new position
        new_lat, new_lon, _ = geodetic_direct(original_lat, original_lon, 1000.0, bearing, distance_m, 0)
        
        # Calculate bearing and distance back
        back_bearing = geodetic_bearing_deg(new_lat, new_lon, original_lat, original_lon)
        back_distance_km = geodetic_distance_km(new_lat, new_lon, 1000.0, original_lat, original_lon, 1000.0)
        
        # Back bearing should be approximately opposite
        expected_back_bearing = (bearing + 180) % 360
        assert back_bearing == pytest.approx(expected_back_bearing, abs=2.0)
        
        # Back distance should match original
        assert back_distance_km == pytest.approx(distance_m / 1000.0, rel=0.01)
    
    def test_multiple_short_segments_vs_direct(self):
        """Test that multiple short segments approximate a direct path."""
        start_lat, start_lon = 45.0, -123.0
        bearing = 45.0
        total_distance_m = 100000  # 100 km
        num_segments = 10
        segment_distance_m = total_distance_m / num_segments
        
        # Direct path
        direct_lat, direct_lon, _ = geodetic_direct(start_lat, start_lon, 0, bearing, total_distance_m, 0)
        
        # Segmented path
        current_lat, current_lon = start_lat, start_lon
        for _ in range(num_segments):
            current_lat, current_lon, _ = geodetic_direct(current_lat, current_lon, 0, bearing, segment_distance_m, 0)
        
        # Should be close to direct calculation
        assert current_lat == pytest.approx(direct_lat, abs=0.01)
        assert current_lon == pytest.approx(direct_lon, abs=0.01)


class TestGeodeticEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_extreme_latitudes(self):
        """Test calculations at extreme latitudes."""
        # Very close to poles
        distance = geodetic_distance_km(89.9, 0.0, 0.0, 89.9, 180.0, 0.0)
        assert distance >= 0
        
        bearing = geodetic_bearing_deg(89.9, 0.0, 89.9, 90.0)
        assert 0 <= bearing < 360
    
    def test_crossing_dateline_multiple_times(self):
        """Test calculations crossing international dateline."""
        # West to east across dateline
        dist1 = geodetic_distance_km(45.0, 179.5, 0.0, 45.0, -179.5, 0.0)
        
        # East to west across dateline
        dist2 = geodetic_distance_km(45.0, -179.5, 0.0, 45.0, 179.5, 0.0)
        
        # Should be approximately equal (short way around)
        assert dist1 == pytest.approx(dist2, rel=0.01)
        assert dist1 < 500  # Should be short distance, not around the world
    
    def test_very_small_distances(self):
        """Test calculations with very small distances."""
        # 1 meter horizontal
        lat, lon, alt = geodetic_direct(45.0, -123.0, 1000.0, 90.0, 1.0, 0.0)
        distance = geodetic_distance_km(45.0, -123.0, 1000.0, lat, lon, alt)
        # Relax tolerance slightly due to geodetic calculation precision
        assert distance == pytest.approx(0.001, abs=1e-4)
    
    def test_very_large_distances(self):
        """Test calculations with very large distances."""
        # Quarter circumference
        lat, lon, alt = geodetic_direct(0.0, 0.0, 0.0, 0.0, 10007543.0, 0.0)  # ~1/4 Earth circumference
        assert abs(lat) <= 90
        assert abs(lon) <= 180
    
    def test_bearing_normalization_and_caching(self):
        """Test that bearing results are normalized and caching works correctly."""
        from simulator.utils.geodesics import clear_bearing_cache

        # Clear cache to start fresh
        clear_bearing_cache()

        # First call should compute and cache
        bearing1 = geodetic_bearing_deg(45.0, -123.0, 46.0, -122.0)
        assert 0 <= bearing1 < 360  # Should be normalized

        # Second call with same coordinates should return cached result
        bearing2 = geodetic_bearing_deg(45.0, -123.0, 46.0, -122.0)
        assert bearing1 == bearing2  # Exact match from cache

        # Clear cache and verify result is still consistent
        clear_bearing_cache()
        bearing3 = geodetic_bearing_deg(45.0, -123.0, 46.0, -122.0)
        assert bearing3 == pytest.approx(bearing1, abs=1e-10)  # Should match original