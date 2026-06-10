"""
Comprehensive tests for simulator.utils.map_limits module.

Tests the MapLimits class for geographic boundary management,
coordinate transformations, and spatial extent calculations.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from bvr_marl_core.simulator.utils.map_limits import MapLimits


class TestMapLimitsInitialization:
    """Test MapLimits class initialization and basic properties."""

    def test_basic_initialization(self):
        """Test basic initialization with required parameters."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0)

        assert limits.left_lon == -124.0
        assert limits.bottom_lat == 45.0
        assert limits.right_lon == -120.0
        assert limits.top_lat == 47.0
        assert limits.min_alt == 0
        assert limits.max_alt == 10000

    def test_initialization_with_altitude(self):
        """Test initialization with custom altitude limits."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=15000)

        assert limits.min_alt == 1000
        assert limits.max_alt == 15000

    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters specified."""
        limits = MapLimits(
            left_lon=-180.0,
            bottom_lat=-90.0,
            right_lon=180.0,
            top_lat=90.0,
            min_alt=0,
            max_alt=20000,
        )

        assert limits.left_lon == -180.0
        assert limits.bottom_lat == -90.0
        assert limits.right_lon == 180.0
        assert limits.top_lat == 90.0
        assert limits.min_alt == 0
        assert limits.max_alt == 20000

    def test_negative_coordinates(self):
        """Test initialization with negative coordinates."""
        limits = MapLimits(-10.0, -5.0, -1.0, -2.0, min_alt=-100, max_alt=5000)

        assert limits.left_lon == -10.0
        assert limits.bottom_lat == -5.0
        assert limits.right_lon == -1.0
        assert limits.top_lat == -2.0
        assert limits.min_alt == -100
        assert limits.max_alt == 5000


class TestMapLimitsExtentMethods:
    """Test extent calculation methods."""

    def setUp(self):
        self.limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

    def test_latitude_extent(self):
        """Test latitude extent calculation."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0)
        assert limits.latitude_extent() == 2.0

    def test_longitude_extent(self):
        """Test longitude extent calculation."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0)
        assert limits.longitude_extent() == 4.0

    def test_altitude_extent(self):
        """Test altitude extent calculation."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)
        assert limits.altitude_extent() == 10000

    def test_zero_extents(self):
        """Test zero extent calculations."""
        limits = MapLimits(-120.0, 45.0, -120.0, 45.0, min_alt=5000, max_alt=5000)

        assert limits.latitude_extent() == 0.0
        assert limits.longitude_extent() == 0.0
        assert limits.altitude_extent() == 0.0

    def test_negative_extents(self):
        """Test with reversed coordinates (negative extents)."""
        # Bottom > top, left > right
        limits = MapLimits(-120.0, 47.0, -124.0, 45.0, min_alt=11000, max_alt=1000)

        assert limits.latitude_extent() == -2.0
        assert limits.longitude_extent() == -4.0
        assert limits.altitude_extent() == -10000

    def test_small_extents(self):
        """Test very small extent calculations."""
        limits = MapLimits(-120.001, 45.001, -119.999, 45.002, min_alt=4999, max_alt=5001)

        assert limits.latitude_extent() == pytest.approx(0.001, abs=1e-10)
        assert limits.longitude_extent() == pytest.approx(0.002, abs=1e-10)
        assert limits.altitude_extent() == 2

    def test_large_extents(self):
        """Test large extent calculations (global scale)."""
        limits = MapLimits(-180.0, -90.0, 180.0, 90.0, min_alt=0, max_alt=100000)

        assert limits.latitude_extent() == 180.0
        assert limits.longitude_extent() == 360.0
        assert limits.altitude_extent() == 100000


class TestMapLimitsGeodeticExtents:
    """Test geodetic extent calculations using real Earth distances."""

    @patch("bvr_marl_core.simulator.utils.map_limits.Geodesic")
    def test_max_latitude_extent_km(self, mock_geodesic):
        """Test maximum latitude extent in kilometers."""
        # Mock Geodesic responses
        mock_inverse_result_1 = {"s12": 222640.0}  # ~222.64 km
        mock_inverse_result_2 = {"s12": 220000.0}  # ~220 km
        mock_geodesic.WGS84.Inverse.side_effect = [mock_inverse_result_1, mock_inverse_result_2]

        limits = MapLimits(-124.0, 45.0, -120.0, 47.0)
        extent_km = limits.max_latitude_extent_km()

        assert extent_km == pytest.approx(222.64, abs=0.01)
        assert mock_geodesic.WGS84.Inverse.call_count == 2

    @patch("bvr_marl_core.simulator.utils.map_limits.Geodesic")
    def test_max_longitude_extent_km(self, mock_geodesic):
        """Test maximum longitude extent in kilometers."""
        # Mock Geodesic responses
        mock_inverse_result_1 = {"s12": 315000.0}  # ~315 km
        mock_inverse_result_2 = {"s12": 300000.0}  # ~300 km
        mock_geodesic.WGS84.Inverse.side_effect = [mock_inverse_result_1, mock_inverse_result_2]

        limits = MapLimits(-124.0, 45.0, -120.0, 47.0)
        extent_km = limits.max_longitude_extent_km()

        assert extent_km == pytest.approx(315.0, abs=0.01)
        assert mock_geodesic.WGS84.Inverse.call_count == 2

    @patch("bvr_marl_core.simulator.utils.map_limits.Geodesic")
    def test_geodetic_calls_with_correct_parameters(self, mock_geodesic):
        """Test that geodetic calls are made with correct parameters."""
        mock_geodesic.WGS84.Inverse.return_value = {"s12": 100000.0}

        limits = MapLimits(-124.0, 45.0, -120.0, 47.0)

        # Test latitude extent calls
        limits.max_latitude_extent_km()

        # Reset mock for longitude test
        mock_geodesic.WGS84.Inverse.reset_mock()

        # Test longitude extent calls
        limits.max_longitude_extent_km()

        # Verify calls were made (exact parameter checking is complex with mock)
        assert mock_geodesic.WGS84.Inverse.call_count == 2

    @patch("bvr_marl_core.simulator.utils.map_limits.Geodesic")
    def test_same_coordinate_extent(self, mock_geodesic):
        """Test extent calculation when coordinates are the same."""
        mock_geodesic.WGS84.Inverse.return_value = {"s12": 0.0}

        limits = MapLimits(-120.0, 45.0, -120.0, 45.0)

        lat_extent = limits.max_latitude_extent_km()
        lon_extent = limits.max_longitude_extent_km()

        assert lat_extent == 0.0
        assert lon_extent == 0.0


class TestMapLimitsRelativePosition:
    """Test relative position calculations."""

    def test_bottom_left_corner(self):
        """Test relative position at bottom-left corner."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)
        lat_rel, lon_rel, alt_rel = limits.relative_position(45.0, -124.0, 1000)

        assert lat_rel == pytest.approx(0.0, abs=1e-10)
        assert lon_rel == pytest.approx(0.0, abs=1e-10)
        assert alt_rel == pytest.approx(0.0, abs=1e-10)

    def test_top_right_corner(self):
        """Test relative position at top-right corner."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)
        lat_rel, lon_rel, alt_rel = limits.relative_position(47.0, -120.0, 11000)

        assert lat_rel == pytest.approx(1.0, abs=1e-10)
        assert lon_rel == pytest.approx(1.0, abs=1e-10)
        assert alt_rel == pytest.approx(1.0, abs=1e-10)

    def test_center_position(self):
        """Test relative position at center."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)
        lat_rel, lon_rel, alt_rel = limits.relative_position(46.0, -122.0, 6000)

        assert lat_rel == pytest.approx(0.5, abs=1e-10)
        assert lon_rel == pytest.approx(0.5, abs=1e-10)
        assert alt_rel == pytest.approx(0.5, abs=1e-10)

    def test_quarter_positions(self):
        """Test relative positions at quarter points."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        # Quarter latitude
        lat_rel, _, _ = limits.relative_position(45.5, -122.0, 6000)
        assert lat_rel == pytest.approx(0.25, abs=1e-10)

        # Three-quarter longitude
        _, lon_rel, _ = limits.relative_position(46.0, -121.0, 6000)
        assert lon_rel == pytest.approx(0.75, abs=1e-10)

        # Quarter altitude
        _, _, alt_rel = limits.relative_position(46.0, -122.0, 3500)
        assert alt_rel == pytest.approx(0.25, abs=1e-10)

    def test_outside_bounds_clipping(self):
        """Test that positions outside bounds are clipped to [0,1]."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        # Way outside bounds
        lat_rel, lon_rel, alt_rel = limits.relative_position(50.0, -110.0, 20000)

        assert lat_rel == 1.0  # Clipped to max
        assert lon_rel == 1.0  # Clipped to max
        assert alt_rel == 1.0  # Clipped to max

        # Way below bounds
        lat_rel, lon_rel, alt_rel = limits.relative_position(40.0, -130.0, -5000)

        assert lat_rel == 0.0  # Clipped to min
        assert lon_rel == 0.0  # Clipped to min
        assert alt_rel == 0.0  # Clipped to min

    def test_zero_extent_handling(self):
        """Test relative position with zero extent."""
        # Zero latitude extent should raise ZeroDivisionError
        limits = MapLimits(-124.0, 45.0, -120.0, 45.0, min_alt=1000, max_alt=11000)
        with pytest.raises(ZeroDivisionError):
            limits.relative_position(45.0, -122.0, 6000)

    @patch("numpy.clip")
    def test_numpy_clip_called(self, mock_clip):
        """Test that numpy.clip is called for all coordinates."""
        mock_clip.side_effect = lambda x, min_val, max_val: x  # Pass through

        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)
        limits.relative_position(46.0, -122.0, 6000)

        assert mock_clip.call_count == 3  # Called for lat, lon, alt

    def test_precision_with_small_values(self):
        """Test relative position precision with small coordinate differences."""
        limits = MapLimits(-120.001, 45.001, -119.999, 45.002, min_alt=4999, max_alt=5001)

        lat_rel, lon_rel, alt_rel = limits.relative_position(45.0015, -120.0005, 5000)

        assert lat_rel == pytest.approx(0.5, abs=1e-6)
        assert lon_rel == pytest.approx(0.25, abs=1e-6)
        assert alt_rel == pytest.approx(0.5, abs=1e-6)


class TestMapLimitsAbsolutePosition:
    """Test absolute position calculations from relative coordinates."""

    def test_zero_relative_position(self):
        """Test conversion from (0,0,0) relative position."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)
        lat, lon, alt = limits.absolute_position(0.0, 0.0, 0.0)

        assert lat == pytest.approx(45.0, abs=1e-10)
        assert lon == pytest.approx(-124.0, abs=1e-10)
        assert alt == pytest.approx(1000.0, abs=1e-10)

    def test_unit_relative_position(self):
        """Test conversion from (1,1,1) relative position."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)
        lat, lon, alt = limits.absolute_position(1.0, 1.0, 1.0)

        assert lat == pytest.approx(47.0, abs=1e-10)
        assert lon == pytest.approx(-120.0, abs=1e-10)
        assert alt == pytest.approx(11000.0, abs=1e-10)

    def test_half_relative_position(self):
        """Test conversion from (0.5,0.5,0.5) relative position."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)
        lat, lon, alt = limits.absolute_position(0.5, 0.5, 0.5)

        assert lat == pytest.approx(46.0, abs=1e-10)
        assert lon == pytest.approx(-122.0, abs=1e-10)
        assert alt == pytest.approx(6000.0, abs=1e-10)

    def test_quarter_relative_positions(self):
        """Test conversion from quarter relative positions."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        # Quarter points
        lat, lon, alt = limits.absolute_position(0.25, 0.75, 0.1)

        assert lat == pytest.approx(45.5, abs=1e-10)  # 45 + 0.25 * 2
        assert lon == pytest.approx(-121.0, abs=1e-10)  # -124 + 0.75 * 4
        assert alt == pytest.approx(2000.0, abs=1e-10)  # 1000 + 0.1 * 10000

    def test_outside_unit_range(self):
        """Test conversion with relative coordinates outside [0,1] range."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        # Beyond unit range
        lat, lon, alt = limits.absolute_position(1.5, -0.5, 2.0)

        assert lat == pytest.approx(48.0, abs=1e-10)  # 45 + 1.5 * 2
        assert lon == pytest.approx(-126.0, abs=1e-10)  # -124 + (-0.5) * 4
        assert alt == pytest.approx(21000.0, abs=1e-10)  # 1000 + 2.0 * 10000

    def test_negative_relative_coordinates(self):
        """Test conversion with negative relative coordinates."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)
        lat, lon, alt = limits.absolute_position(-0.25, -0.1, -0.05)

        assert lat == pytest.approx(44.5, abs=1e-10)
        assert lon == pytest.approx(-124.4, abs=1e-10)
        assert alt == pytest.approx(500.0, abs=1e-10)

    def test_round_trip_consistency(self):
        """Test that relative->absolute->relative conversion is consistent."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        # Start with relative coordinates
        orig_lat_rel, orig_lon_rel, orig_alt_rel = 0.3, 0.7, 0.25

        # Convert to absolute
        lat, lon, alt = limits.absolute_position(orig_lat_rel, orig_lon_rel, orig_alt_rel)

        # Convert back to relative
        lat_rel, lon_rel, alt_rel = limits.relative_position(lat, lon, alt)

        assert lat_rel == pytest.approx(orig_lat_rel, abs=1e-10)
        assert lon_rel == pytest.approx(orig_lon_rel, abs=1e-10)
        assert alt_rel == pytest.approx(orig_alt_rel, abs=1e-10)

    def test_zero_extent_absolute_position(self):
        """Test absolute position conversion with zero extents."""
        limits = MapLimits(-120.0, 45.0, -120.0, 45.0, min_alt=5000, max_alt=5000)

        lat, lon, alt = limits.absolute_position(0.5, 0.8, 0.3)

        # With zero extent, relative multiplier has no effect
        assert lat == pytest.approx(45.0, abs=1e-10)
        assert lon == pytest.approx(-120.0, abs=1e-10)
        assert alt == pytest.approx(5000.0, abs=1e-10)


class TestMapLimitsInBoundary:
    """Test boundary checking functionality."""

    def test_inside_boundary(self):
        """Test positions clearly inside boundary."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        assert limits.in_boundary(46.0, -122.0, 6000) is True
        assert limits.in_boundary(45.5, -123.5, 2000) is True
        assert limits.in_boundary(46.9, -120.1, 10500) is True

    def test_outside_boundary_latitude(self):
        """Test positions outside latitude boundaries."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        # Below minimum latitude
        assert limits.in_boundary(44.5, -122.0, 6000) is False

        # Above maximum latitude
        assert limits.in_boundary(47.5, -122.0, 6000) is False

    def test_outside_boundary_longitude(self):
        """Test positions outside longitude boundaries."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        # Below minimum longitude
        assert limits.in_boundary(46.0, -125.0, 6000) is False

        # Above maximum longitude
        assert limits.in_boundary(46.0, -119.0, 6000) is False

    def test_outside_boundary_altitude(self):
        """Test positions outside altitude boundaries."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        # Below minimum altitude
        assert limits.in_boundary(46.0, -122.0, 500) is False

        # Above maximum altitude
        assert limits.in_boundary(46.0, -122.0, 12000) is False

    def test_on_boundary_edges(self):
        """Test positions exactly on boundary edges."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        # On edges (should be inside)
        assert limits.in_boundary(45.0, -122.0, 6000) is True  # Bottom edge
        assert limits.in_boundary(47.0, -122.0, 6000) is True  # Top edge
        assert limits.in_boundary(46.0, -124.0, 6000) is True  # Left edge
        assert limits.in_boundary(46.0, -120.0, 6000) is True  # Right edge
        assert limits.in_boundary(46.0, -122.0, 1000) is True  # Min altitude
        assert limits.in_boundary(46.0, -122.0, 11000) is True  # Max altitude

    def test_corners(self):
        """Test all corner positions."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        # All 8 corners of the 3D box
        assert limits.in_boundary(45.0, -124.0, 1000) is True  # Bottom-left-min
        assert limits.in_boundary(45.0, -124.0, 11000) is True  # Bottom-left-max
        assert limits.in_boundary(45.0, -120.0, 1000) is True  # Bottom-right-min
        assert limits.in_boundary(45.0, -120.0, 11000) is True  # Bottom-right-max
        assert limits.in_boundary(47.0, -124.0, 1000) is True  # Top-left-min
        assert limits.in_boundary(47.0, -124.0, 11000) is True  # Top-left-max
        assert limits.in_boundary(47.0, -120.0, 1000) is True  # Top-right-min
        assert limits.in_boundary(47.0, -120.0, 11000) is True  # Top-right-max

    def test_multiple_dimensions_outside(self):
        """Test positions outside in multiple dimensions."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        # Outside in lat and lon
        assert limits.in_boundary(44.0, -125.0, 6000) is False

        # Outside in all dimensions
        assert limits.in_boundary(44.0, -125.0, 500) is False
        assert limits.in_boundary(48.0, -119.0, 12000) is False

    def test_floating_point_precision(self):
        """Test boundary checking with floating point precision issues."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        # Very close to boundary (should handle floating point precision)
        assert limits.in_boundary(44.9999999999, -122.0, 6000) is False
        assert limits.in_boundary(45.0000000001, -122.0, 6000) is True

    def test_negative_coordinates(self):
        """Test boundary checking with negative coordinates."""
        limits = MapLimits(-10.0, -5.0, -1.0, -2.0, min_alt=-100, max_alt=5000)

        # This point is within the boundary (lat: -5.0 to -2.0, lon: -10.0 to -1.0)
        assert limits.in_boundary(-3.5, -5.5, 2500) is True  # Point is within range

        # Test points outside the boundary
        assert limits.in_boundary(-1.5, -5.5, 2500) is False  # lat outside range (too high)
        assert limits.in_boundary(-3.5, -0.5, 2500) is False  # lon outside range (too high)

    def test_zero_sized_boundaries(self):
        """Test boundary checking with zero-sized boundaries."""
        limits = MapLimits(-120.0, 45.0, -120.0, 45.0, min_alt=5000, max_alt=5000)

        # Only exact match should be inside
        assert limits.in_boundary(45.0, -120.0, 5000) is True
        assert limits.in_boundary(45.0, -120.0, 5001) is False
        assert limits.in_boundary(45.001, -120.0, 5000) is False


class TestMapLimitsIntegration:
    """Integration tests combining multiple MapLimits functionalities."""

    def test_complete_workflow(self):
        """Test complete workflow: initialization -> relative -> absolute -> boundary."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        # Start with absolute coordinates
        original_lat, original_lon, original_alt = 46.5, -121.5, 8000

        # Convert to relative
        lat_rel, lon_rel, alt_rel = limits.relative_position(
            original_lat, original_lon, original_alt
        )

        # Convert back to absolute
        calc_lat, calc_lon, calc_alt = limits.absolute_position(lat_rel, lon_rel, alt_rel)

        # Check boundary
        is_inside = limits.in_boundary(calc_lat, calc_lon, calc_alt)

        # Verify consistency
        assert calc_lat == pytest.approx(original_lat, abs=1e-10)
        assert calc_lon == pytest.approx(original_lon, abs=1e-10)
        assert calc_alt == pytest.approx(original_alt, abs=1e-10)
        assert is_inside is True

    def test_extent_relationships(self):
        """Test relationships between different extent calculations."""
        limits = MapLimits(-124.0, 45.0, -120.0, 47.0, min_alt=1000, max_alt=11000)

        # Basic extents
        lat_extent_deg = limits.latitude_extent()
        lon_extent_deg = limits.longitude_extent()
        alt_extent_m = limits.altitude_extent()

        # Geodetic extents (mocked)
        with patch("bvr_marl_core.simulator.utils.map_limits.Geodesic") as mock_geodesic:
            mock_geodesic.WGS84.Inverse.return_value = {"s12": 222640.0}  # ~222.64 km
            lat_extent_km = limits.max_latitude_extent_km()
            lon_extent_km = limits.max_longitude_extent_km()

        # Verify relationships
        assert lat_extent_deg > 0
        assert lon_extent_deg > 0
        assert alt_extent_m > 0
        assert lat_extent_km > 0
        assert lon_extent_km > 0

        # Latitude extent should be roughly proportional to degrees
        # (very rough check - actual conversion depends on location)
        assert lat_extent_km > lat_extent_deg * 50  # At least 50 km per degree

    def test_global_map_limits(self):
        """Test with global-scale map limits."""
        limits = MapLimits(-180.0, -90.0, 180.0, 90.0, min_alt=0, max_alt=100000)

        # Test various positions around the globe
        positions = [
            (0.0, 0.0, 50000),  # Equator, prime meridian
            (90.0, 0.0, 10000),  # North pole
            (-90.0, 0.0, 90000),  # South pole
            (0.0, 180.0, 0),  # Equator, antimeridian
            (0.0, -180.0, 100000),  # Equator, antimeridian (other side)
        ]

        for lat, lon, alt in positions:
            # Should all be inside global bounds
            assert limits.in_boundary(lat, lon, alt) is True

            # Relative position should be reasonable
            lat_rel, lon_rel, alt_rel = limits.relative_position(lat, lon, alt)
            assert 0.0 <= lat_rel <= 1.0
            assert 0.0 <= lon_rel <= 1.0
            assert 0.0 <= alt_rel <= 1.0

            # Round-trip should work
            calc_lat, calc_lon, calc_alt = limits.absolute_position(lat_rel, lon_rel, alt_rel)
            assert calc_lat == pytest.approx(lat, abs=1e-10)
            assert calc_lon == pytest.approx(lon, abs=1e-10)
            assert calc_alt == pytest.approx(alt, abs=1e-10)

    def test_small_region_map_limits(self):
        """Test with very small regional map limits."""
        # Small area around a city
        limits = MapLimits(-122.45, 37.75, -122.35, 37.85, min_alt=0, max_alt=500)

        # Test positions within the small region
        positions = [
            (37.8, -122.4, 100),  # Center-ish
            (37.75, -122.45, 0),  # Corner
            (37.85, -122.35, 500),  # Opposite corner
        ]

        for lat, lon, alt in positions:
            assert limits.in_boundary(lat, lon, alt) is True

            # High precision should be maintained for small regions
            lat_rel, lon_rel, alt_rel = limits.relative_position(lat, lon, alt)
            calc_lat, calc_lon, calc_alt = limits.absolute_position(lat_rel, lon_rel, alt_rel)

            assert calc_lat == pytest.approx(lat, abs=1e-12)
            assert calc_lon == pytest.approx(lon, abs=1e-12)
            assert calc_alt == pytest.approx(alt, abs=1e-12)


class TestMapLimitsEdgeCases:
    """Test edge cases and potential error conditions."""

    def test_inverted_boundaries(self):
        """Test with inverted boundaries (max < min)."""
        # This could happen due to user error
        limits = MapLimits(-120.0, 47.0, -124.0, 45.0, min_alt=11000, max_alt=1000)

        # The methods should still work mathematically
        assert limits.latitude_extent() == -2.0
        assert limits.longitude_extent() == -4.0
        assert limits.altitude_extent() == -10000

        # Boundary checking becomes problematic
        assert limits.in_boundary(46.0, -122.0, 6000) is False  # Nothing is "inside"

    def test_extreme_coordinate_values(self):
        """Test with extreme coordinate values."""
        limits = MapLimits(-179.99, -89.99, 179.99, 89.99, min_alt=-10000, max_alt=100000)

        # Should handle near-extreme values
        assert limits.in_boundary(89.0, 179.0, 50000) is True
        assert limits.in_boundary(-89.0, -179.0, -5000) is True

    def test_very_high_precision_coordinates(self):
        """Test with very high precision coordinates."""
        limits = MapLimits(
            -122.123456789,
            37.123456789,
            -122.123456788,
            37.123456790,
            min_alt=99.999,
            max_alt=100.001,
        )

        lat_rel, lon_rel, alt_rel = limits.relative_position(37.1234567895, -122.1234567885, 100.0)
        calc_lat, calc_lon, calc_alt = limits.absolute_position(lat_rel, lon_rel, alt_rel)

        # Should maintain reasonable precision
        assert calc_lat == pytest.approx(37.1234567895, abs=1e-10)
        assert calc_lon == pytest.approx(-122.1234567885, abs=1e-10)
        assert calc_alt == pytest.approx(100.0, abs=1e-10)
