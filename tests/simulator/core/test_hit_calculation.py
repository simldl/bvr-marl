"""
Comprehensive tests for simulator.core.hit_calculation module.

Tests the MissileHitCalculator class for continuous collision detection,
coordinate transformations, and missile-target interaction calculations.
"""

import math
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from bvr_marl_core.simulator.core.hit_calculation import (
    HitParams,
    MissileHitCalculator,
    _closest_approach_time,
    _latlonalt_to_local_m,
)


class TestHitParams:
    """Test HitParams dataclass."""

    def test_default_initialization(self):
        """Test HitParams with default values."""
        params = HitParams()

        assert params.fuse_radius_m == 500.0
        assert params.target_radius_m == 0.0
        assert params.max_consider_range_m == 50000.0

    def test_custom_initialization(self):
        """Test HitParams with custom values."""
        params = HitParams(fuse_radius_m=100.0, target_radius_m=20.0, max_consider_range_m=25000.0)

        assert params.fuse_radius_m == 100.0
        assert params.target_radius_m == 20.0
        assert params.max_consider_range_m == 25000.0

    def test_dataclass_equality(self):
        """Test HitParams equality comparison."""
        params1 = HitParams(fuse_radius_m=100.0, target_radius_m=10.0)
        params2 = HitParams(fuse_radius_m=100.0, target_radius_m=10.0)
        params3 = HitParams(fuse_radius_m=200.0, target_radius_m=10.0)

        assert params1 == params2
        assert params1 != params3


class TestLatLonAltToLocalM:
    """Test _latlonalt_to_local_m coordinate conversion function."""

    def test_origin_conversion(self):
        """Test conversion at the origin point."""
        result = _latlonalt_to_local_m(45.0, -123.0, 1000.0, 45.0, -123.0)
        expected = np.array([0.0, 0.0, 1000.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_latitude_conversion(self):
        """Test conversion with latitude difference."""
        # 1 degree latitude ≈ 111,320 meters
        result = _latlonalt_to_local_m(46.0, -123.0, 0.0, 45.0, -123.0)

        assert result[0] == pytest.approx(0.0, abs=1e-10)  # x (longitude)
        assert result[1] == pytest.approx(111320.0, abs=1.0)  # y (latitude)
        assert result[2] == pytest.approx(0.0, abs=1e-10)  # z (altitude)

    def test_longitude_conversion(self):
        """Test conversion with longitude difference."""
        # Longitude scaling depends on latitude (cos factor)
        lat_ref = 45.0
        expected_scale = 111320.0 * math.cos(math.radians(lat_ref))

        result = _latlonalt_to_local_m(45.0, -122.0, 0.0, lat_ref, -123.0)

        assert result[0] == pytest.approx(expected_scale, rel=0.01)  # x (longitude)
        assert result[1] == pytest.approx(0.0, abs=1e-10)  # y (latitude)
        assert result[2] == pytest.approx(0.0, abs=1e-10)  # z (altitude)

    def test_altitude_conversion(self):
        """Test conversion with altitude difference."""
        result = _latlonalt_to_local_m(45.0, -123.0, 2000.0, 45.0, -123.0)
        expected = np.array([0.0, 0.0, 2000.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_combined_conversion(self):
        """Test conversion with all coordinate differences."""
        lat_ref, lon_ref = 45.0, -123.0
        lat_scale = 111320.0
        lon_scale = 111320.0 * math.cos(math.radians(lat_ref))

        result = _latlonalt_to_local_m(46.0, -122.0, 1500.0, lat_ref, lon_ref)

        assert result[0] == pytest.approx(lon_scale, rel=0.01)  # 1° longitude
        assert result[1] == pytest.approx(lat_scale, rel=0.01)  # 1° latitude
        assert result[2] == pytest.approx(1500.0, abs=1e-10)  # altitude

    def test_negative_coordinates(self):
        """Test conversion with negative coordinate differences."""
        result = _latlonalt_to_local_m(44.0, -124.0, 500.0, 45.0, -123.0)

        # Should have negative x and y components
        assert result[0] < 0  # West
        assert result[1] < 0  # South
        assert result[2] == pytest.approx(500.0, abs=1e-10)

    def test_equatorial_longitude_scaling(self):
        """Test longitude scaling at equator (cos(0) = 1)."""
        result = _latlonalt_to_local_m(0.0, 1.0, 0.0, 0.0, 0.0)

        # At equator, longitude scale should be full 111320
        assert result[0] == pytest.approx(111320.0, rel=0.01)

    def test_polar_longitude_scaling(self):
        """Test longitude scaling near poles."""
        # Near north pole, longitude scaling should be very small
        result = _latlonalt_to_local_m(89.0, 1.0, 0.0, 89.0, 0.0)

        # cos(89°) is very small
        expected_scale = 111320.0 * math.cos(math.radians(89.0))
        assert result[0] == pytest.approx(expected_scale, rel=0.01)
        assert result[0] < 2000  # Should be much smaller than equatorial

    def test_return_type(self):
        """Test that function returns numpy array with correct dtype."""
        result = _latlonalt_to_local_m(45.0, -123.0, 1000.0, 45.0, -123.0)

        assert isinstance(result, np.ndarray)
        assert result.dtype == float
        assert result.shape == (3,)


class TestClosestApproachTime:
    """Test _closest_approach_time function."""

    def test_zero_relative_velocity(self):
        """Test with zero relative velocity."""
        r0 = np.array([10.0, 5.0, 2.0])
        v_rel = np.array([0.0, 0.0, 0.0])

        result = _closest_approach_time(r0, v_rel)
        assert result == 0.0

    def test_perpendicular_motion(self):
        """Test with perpendicular initial position and velocity."""
        r0 = np.array([10.0, 0.0, 0.0])  # Initially offset in x
        v_rel = np.array([0.0, 1.0, 0.0])  # Moving in y direction

        result = _closest_approach_time(r0, v_rel)
        # r0 · v_rel = 0, so t* = 0
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_approaching_motion(self):
        """Test with approaching motion (negative t*)."""
        r0 = np.array([10.0, 0.0, 0.0])  # Initially separated
        v_rel = np.array([-2.0, 0.0, 0.0])  # Approaching

        result = _closest_approach_time(r0, v_rel)
        # t* = -r0·v_rel / ||v_rel||² = -10*(-2) / 4 = 5
        expected = 5.0
        assert result == pytest.approx(expected, abs=1e-10)

    def test_separating_motion(self):
        """Test with separating motion (positive t*)."""
        r0 = np.array([10.0, 0.0, 0.0])  # Initially separated
        v_rel = np.array([2.0, 0.0, 0.0])  # Separating

        result = _closest_approach_time(r0, v_rel)
        # t* = -r0·v_rel / ||v_rel||² = -10*2 / 4 = -5
        expected = -5.0
        assert result == pytest.approx(expected, abs=1e-10)

    def test_3d_motion(self):
        """Test with 3D motion vectors."""
        r0 = np.array([3.0, 4.0, 0.0])  # Initial separation
        v_rel = np.array([1.0, -2.0, 1.0])  # 3D relative velocity

        dot_product = np.dot(r0, v_rel)  # 3*1 + 4*(-2) + 0*1 = -5
        v_rel_squared = np.dot(v_rel, v_rel)  # 1² + (-2)² + 1² = 6
        expected = -dot_product / v_rel_squared  # 5/6

        result = _closest_approach_time(r0, v_rel)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_very_small_velocity(self):
        """Test with very small but non-zero velocity."""
        r0 = np.array([10.0, 0.0, 0.0])
        v_rel = np.array([1e-15, 0.0, 0.0])  # Very small velocity

        result = _closest_approach_time(r0, v_rel)
        # Should handle numerically small values gracefully
        assert isinstance(result, float)
        assert not math.isnan(result)

    def test_float_conversion(self):
        """Test that result is converted to Python float."""
        r0 = np.array([1.0, 1.0, 1.0])
        v_rel = np.array([1.0, 1.0, 1.0])

        result = _closest_approach_time(r0, v_rel)

        assert isinstance(result, float)
        assert not isinstance(result, np.floating)


class TestMissileHitCalculatorInitialization:
    """Test MissileHitCalculator initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        calc = MissileHitCalculator()

        assert calc.params.fuse_radius_m == 500.0
        assert calc.params.target_radius_m == 0.0
        assert calc.params.max_consider_range_m == 50000.0
        assert calc._preposes == {}
        assert calc._enu_origin is None

    def test_custom_params_initialization(self):
        """Test initialization with custom HitParams."""
        params = HitParams(fuse_radius_m=100.0, target_radius_m=25.0)
        calc = MissileHitCalculator(params)

        assert calc.params is params
        assert calc.params.fuse_radius_m == 100.0
        assert calc.params.target_radius_m == 25.0

    def test_none_params_initialization(self):
        """Test initialization with None params creates default."""
        calc = MissileHitCalculator(None)

        assert isinstance(calc.params, HitParams)
        assert calc.params.fuse_radius_m == 500.0


class TestMissileHitCalculatorTickSnapshot:
    """Test begin_tick_snapshot method."""

    def test_empty_units_snapshot(self):
        """Test snapshot with empty units iterable."""
        calc = MissileHitCalculator()

        calc.begin_tick_snapshot([])

        assert calc._preposes == {}
        assert calc._enu_origin is None

    def test_single_unit_snapshot(self):
        """Test snapshot with single unit."""
        calc = MissileHitCalculator()

        # Mock unit
        unit = Mock()
        unit.id = 1
        unit.position = Mock()
        unit.position.lat = 45.0
        unit.position.lon = -123.0
        unit.position.alt = 1000.0

        calc.begin_tick_snapshot([unit])

        assert calc._preposes[1] == (45.0, -123.0, 1000.0)
        assert calc._enu_origin == (45.0, -123.0)

    def test_multiple_units_snapshot(self):
        """Test snapshot with multiple units."""
        calc = MissileHitCalculator()

        # Mock units
        units = []
        for i in range(3):
            unit = Mock()
            unit.id = i + 1
            unit.position = Mock()
            unit.position.lat = 45.0 + i
            unit.position.lon = -123.0 - i
            unit.position.alt = 1000.0 + i * 100
            units.append(unit)

        calc.begin_tick_snapshot(units)

        # Should store all positions
        assert len(calc._preposes) == 3
        assert calc._preposes[1] == (45.0, -123.0, 1000.0)
        assert calc._preposes[2] == (46.0, -124.0, 1100.0)
        assert calc._preposes[3] == (47.0, -125.0, 1200.0)

        # Origin should be first unit's position
        assert calc._enu_origin == (45.0, -123.0)

    def test_unit_without_position(self):
        """Test snapshot with unit that has no position."""
        calc = MissileHitCalculator()

        # Mock unit without position
        unit = Mock()
        unit.id = 1
        unit.position = None

        calc.begin_tick_snapshot([unit])

        assert calc._preposes == {}
        assert calc._enu_origin is None

    def test_unit_without_position_attribute(self):
        """Test snapshot with unit that has no position attribute."""
        calc = MissileHitCalculator()

        # Mock unit without position attribute
        unit = Mock(spec=[])  # Explicitly exclude 'position' attribute
        unit.id = 1

        calc.begin_tick_snapshot([unit])

        assert calc._preposes == {}
        assert calc._enu_origin is None

    def test_snapshot_clears_previous_data(self):
        """Test that snapshot clears previous tick data."""
        calc = MissileHitCalculator()
        calc._preposes[999] = (0.0, 0.0, 0.0)  # Old data
        calc._enu_origin = (0.0, 0.0)  # Old origin

        # Mock new unit
        unit = Mock()
        unit.id = 1
        unit.position = Mock()
        unit.position.lat = 45.0
        unit.position.lon = -123.0
        unit.position.alt = 1000.0

        calc.begin_tick_snapshot([unit])

        # Old data should be cleared
        assert 999 not in calc._preposes
        assert calc._preposes[1] == (45.0, -123.0, 1000.0)
        assert calc._enu_origin == (45.0, -123.0)

    def test_float_conversion(self):
        """Test that coordinates are converted to float."""
        calc = MissileHitCalculator()

        # Mock unit with non-float coordinates
        unit = Mock()
        unit.id = 1
        unit.position = Mock()
        unit.position.lat = np.float64(45.0)  # NumPy float
        unit.position.lon = -123  # Integer
        unit.position.alt = "1000.0"  # String

        calc.begin_tick_snapshot([unit])

        # Should convert all to Python float
        stored = calc._preposes[1]
        assert all(isinstance(x, float) for x in stored)
        assert stored == (45.0, -123.0, 1000.0)


class MockUnit:
    """Helper class for creating mock units with positions."""

    def __init__(
        self, unit_id: int, lat: float, lon: float, alt: float, is_destroyed: bool = False
    ):
        self.id = unit_id
        self.position = Mock()
        self.position.lat = lat
        self.position.lon = lon
        self.position.alt = alt
        self.is_destroyed = is_destroyed


class TestMissileHitCalculatorHitDetection:
    """Test check_and_maybe_hit method."""

    def test_no_targets_no_hit(self):
        """Test with no potential targets."""
        calc = MissileHitCalculator()

        missile = MockUnit(1, 45.0, -123.0, 1000.0)

        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [], 1.0)

        assert hit is False
        assert victim is None
        assert t_frac is None

    def test_missile_as_target_ignored(self):
        """Test that missile is ignored when in targets list."""
        calc = MissileHitCalculator()

        missile = MockUnit(1, 45.0, -123.0, 1000.0)

        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [missile], 1.0)

        assert hit is False
        assert victim is None
        assert t_frac is None

    def test_destroyed_target_ignored(self):
        """Test that destroyed targets are ignored."""
        calc = MissileHitCalculator()

        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 45.0, -123.0, 1000.0, is_destroyed=True)

        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target], 1.0)

        assert hit is False
        assert victim is None
        assert t_frac is None

    def test_target_without_is_destroyed_attribute(self):
        """Test target without is_destroyed attribute (should not be ignored)."""
        calc = MissileHitCalculator()

        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 45.0, -123.0, 1000.0)
        # Remove is_destroyed attribute to simulate unit without this attribute
        delattr(target, "is_destroyed")

        # Should not crash and should consider the target
        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target], 1.0)

        # Close enough for hit with default fuse radius
        assert hit is True
        assert victim is target

    def test_direct_hit_stationary_targets(self):
        """Test direct hit with stationary overlapping targets."""
        calc = MissileHitCalculator()

        # Both at same position - should hit
        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 45.0, -123.0, 1000.0)

        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target], 1.0)

        assert hit is True
        assert victim is target
        assert t_frac == pytest.approx(0.0, abs=1e-6)  # Immediate hit

    def test_miss_far_apart_stationary(self):
        """Test miss with far apart stationary targets."""
        calc = MissileHitCalculator(HitParams(fuse_radius_m=10.0))  # Small fuse radius

        # Far apart positions
        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 46.0, -122.0, 1000.0)  # ~100km away

        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target], 1.0)

        assert hit is False
        assert victim is None
        assert t_frac is None

    def test_early_reject_filter(self):
        """Test early reject filter with max_consider_range."""
        calc = MissileHitCalculator(HitParams(max_consider_range_m=1000.0))

        # Create positions > 1km apart
        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 45.1, -122.9, 1000.0)  # > 1km away

        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target], 1.0)

        assert hit is False  # Should be rejected early

    @patch(
        "bvr_marl_core.simulator.core.hit_calculation.MissileHitCalculator._compute_motion_vectors"
    )
    def test_motion_vector_computation_called(self, mock_compute):
        """Test that motion vectors are computed for missile and targets."""
        # Mock motion vectors
        missile_vectors = (
            np.array([0.0, 0.0, 0.0]),  # p_m0
            np.array([100.0, 0.0, 0.0]),  # p_m1
            np.array([100.0, 0.0, 0.0]),  # v_m
        )
        target_vectors = (
            np.array([200.0, 0.0, 0.0]),  # p_t0
            np.array([200.0, 0.0, 0.0]),  # p_t1
            np.array([0.0, 0.0, 0.0]),  # v_t
        )

        mock_compute.side_effect = [missile_vectors, target_vectors]

        calc = MissileHitCalculator()
        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 45.0, -123.0, 1000.0)

        calc.check_and_maybe_hit(missile, [target], 1.0)

        # Should call _compute_motion_vectors for both missile and target
        assert mock_compute.call_count == 2

    def test_hit_with_fuse_and_target_radius(self):
        """Test hit calculation with both fuse and target radius."""
        params = HitParams(fuse_radius_m=100.0, target_radius_m=50.0)
        calc = MissileHitCalculator(params)

        # Positions 120m apart - should hit with combined radius 150m
        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 45.001, -123.0, 1000.0)  # ~111m north

        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target], 1.0)

        assert hit is True
        assert victim is target

    def test_first_hit_wins(self):
        """Test that first hit in targets list wins."""
        calc = MissileHitCalculator()

        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target1 = MockUnit(2, 45.0, -123.0, 1000.0)  # Direct hit
        target2 = MockUnit(3, 45.0, -123.0, 1000.0)  # Also direct hit

        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target1, target2], 1.0)

        assert hit is True
        assert victim is target1  # First one wins

    def test_no_snapshot_degrades_gracefully(self):
        """Test that calculation works without begin_tick_snapshot."""
        calc = MissileHitCalculator()
        # Don't call begin_tick_snapshot

        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 45.0, -123.0, 1000.0)

        # Should degrade to discrete check
        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target], 1.0)

        assert hit is True
        assert victim is target


class TestMissileHitCalculatorPrivateMethods:
    """Test private helper methods of MissileHitCalculator."""

    def test_ensure_enu_origin_with_existing_origin(self):
        """Test _ensure_enu_origin when origin already exists."""
        calc = MissileHitCalculator()
        calc._enu_origin = (50.0, -120.0)

        missile = MockUnit(1, 45.0, -123.0, 1000.0)

        calc._ensure_enu_origin(missile)

        # Should not change existing origin
        assert calc._enu_origin == (50.0, -120.0)

    def test_ensure_enu_origin_without_existing_origin(self):
        """Test _ensure_enu_origin when no origin exists."""
        calc = MissileHitCalculator()
        assert calc._enu_origin is None

        missile = MockUnit(1, 45.0, -123.0, 1000.0)

        calc._ensure_enu_origin(missile)

        # Should set origin to missile position
        assert calc._enu_origin == (45.0, -123.0)

    def test_get_t0_pose_existing(self):
        """Test _get_t0_pose with existing data."""
        calc = MissileHitCalculator()
        calc._preposes[1] = (45.0, -123.0, 1000.0)

        unit = MockUnit(1, 46.0, -122.0, 1100.0)  # Current position different

        result = calc._get_t0_pose(unit)

        assert result == (45.0, -123.0, 1000.0)  # Should return stored t0 pose

    def test_get_t0_pose_missing(self):
        """Test _get_t0_pose with missing data."""
        calc = MissileHitCalculator()

        unit = MockUnit(1, 45.0, -123.0, 1000.0)

        result = calc._get_t0_pose(unit)

        assert result is None

    def test_current_pose(self):
        """Test _current_pose extraction."""
        calc = MissileHitCalculator()

        unit = MockUnit(1, 45.5, -123.5, 1500.0)

        result = calc._current_pose(unit)

        assert result == (45.5, -123.5, 1500.0)
        assert all(isinstance(x, float) for x in result)

    def test_compute_motion_vectors_with_t0_data(self):
        """Test _compute_motion_vectors with t0 snapshot data."""
        calc = MissileHitCalculator()
        calc._preposes[1] = (45.0, -123.0, 1000.0)  # t0 position

        unit = MockUnit(1, 45.01, -122.99, 1100.0)  # t1 position (moved)

        with patch(
            "bvr_marl_core.simulator.core.hit_calculation._latlonalt_to_local_m"
        ) as mock_convert:
            mock_convert.side_effect = [
                np.array([0.0, 0.0, 1000.0]),  # t0 position
                np.array([100.0, 1000.0, 1100.0]),  # t1 position
            ]

            p0, p1, v = calc._compute_motion_vectors(unit, 45.0, -123.0)

            assert mock_convert.call_count == 2
            expected_v = np.array([100.0, 1000.0, 100.0])  # p1 - p0
            np.testing.assert_array_equal(v, expected_v)

    def test_compute_motion_vectors_without_t0_data(self):
        """Test _compute_motion_vectors without t0 snapshot data."""
        calc = MissileHitCalculator()
        # No t0 data stored

        unit = MockUnit(1, 45.0, -123.0, 1000.0)

        with patch(
            "bvr_marl_core.simulator.core.hit_calculation._latlonalt_to_local_m"
        ) as mock_convert:
            mock_convert.return_value = np.array([0.0, 0.0, 1000.0])

            p0, p1, v = calc._compute_motion_vectors(unit, 45.0, -123.0)

            # Should call convert twice with same position (degrade to discrete)
            assert mock_convert.call_count == 2
            expected_v = np.array([0.0, 0.0, 0.0])  # No movement
            np.testing.assert_array_equal(v, expected_v)

    def test_early_reject_far_apart(self):
        """Test _early_reject with far apart positions."""
        p_m0 = np.array([0.0, 0.0, 0.0])
        p_m1 = np.array([100.0, 0.0, 0.0])
        p_t0 = np.array([60000.0, 0.0, 0.0])  # 60km away
        p_t1 = np.array([61000.0, 0.0, 0.0])
        max_r = 50000.0

        result = MissileHitCalculator._early_reject(p_m0, p_m1, p_t0, p_t1, max_r)

        assert result is True  # Should be rejected

    def test_early_reject_close_enough(self):
        """Test _early_reject with close enough positions."""
        p_m0 = np.array([0.0, 0.0, 0.0])
        p_m1 = np.array([100.0, 0.0, 0.0])
        p_t0 = np.array([30000.0, 0.0, 0.0])  # 30km away
        p_t1 = np.array([25000.0, 0.0, 0.0])  # Getting closer
        max_r = 50000.0

        result = MissileHitCalculator._early_reject(p_m0, p_m1, p_t0, p_t1, max_r)

        assert result is False  # Should not be rejected

    def test_relative_motion_calculation(self):
        """Test _relative_motion calculation."""
        p_m0 = np.array([10.0, 5.0, 0.0])
        v_m = np.array([100.0, 0.0, 10.0])
        p_t0 = np.array([50.0, 5.0, 0.0])
        v_t = np.array([0.0, 50.0, 0.0])

        r0, v_rel = MissileHitCalculator._relative_motion(p_m0, v_m, p_t0, v_t)

        expected_r0 = np.array([-40.0, 0.0, 0.0])  # p_m0 - p_t0
        expected_v_rel = np.array([100.0, -50.0, 10.0])  # v_m - v_t

        np.testing.assert_array_equal(r0, expected_r0)
        np.testing.assert_array_equal(v_rel, expected_v_rel)

    def test_clamped_closest_time(self):
        """Test _clamped_closest_time clamping behavior."""
        r0 = np.array([10.0, 0.0, 0.0])

        # Test clamping to 0
        v_rel_separating = np.array([5.0, 0.0, 0.0])  # Moving away
        t1 = MissileHitCalculator._clamped_closest_time(r0, v_rel_separating)
        assert t1 == 0.0

        # Test clamping to 1
        v_rel_approaching_far = np.array([-1.0, 0.0, 0.0])  # Slow approach
        t2 = MissileHitCalculator._clamped_closest_time(r0, v_rel_approaching_far)
        # t* = 10 (would be > 1), so clamped to 1
        assert t2 == 1.0

        # Test within [0,1] range
        v_rel_good = np.array([-20.0, 0.0, 0.0])  # Fast approach
        t3 = MissileHitCalculator._clamped_closest_time(r0, v_rel_good)
        # t* = 0.5, within range
        assert t3 == pytest.approx(0.5, abs=1e-10)

    def test_distance_at_time(self):
        """Test _distance_at_time calculation."""
        r0 = np.array([3.0, 4.0, 0.0])  # Initial distance 5
        v_rel = np.array([-6.0, -8.0, 0.0])  # Velocity toward origin

        # At t=0, distance should be 5
        d0 = MissileHitCalculator._distance_at_time(r0, v_rel, 0.0)
        assert d0 == pytest.approx(5.0, abs=1e-10)

        # At t=0.5, should be at origin (distance 0)
        d_half = MissileHitCalculator._distance_at_time(r0, v_rel, 0.5)
        assert d_half == pytest.approx(0.0, abs=1e-10)

        # At t=1, should be at (-3, -4, 0), distance 5 again
        d1 = MissileHitCalculator._distance_at_time(r0, v_rel, 1.0)
        assert d1 == pytest.approx(5.0, abs=1e-10)


class TestMissileHitCalculatorWithCountermeasures:
    """Test hit calculation with countermeasures present."""

    def test_countermeasure_not_detected_as_target(self):
        """Test that countermeasures are not detected as hit targets."""
        calc = MissileHitCalculator(HitParams(fuse_radius_m=500.0))

        # Create missile and target
        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 45.0, -123.0, 1000.0)  # Same position as missile

        # Create countermeasure at same position
        countermeasure = MockUnit(3, 45.0, -123.0, 1000.0)
        countermeasure.is_countermeasure = True
        countermeasure.is_destroyed = False

        # Take snapshot
        calc.begin_tick_snapshot([missile, target, countermeasure])

        # Check hit against target (not countermeasure)
        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target], 1.0)

        assert hit is True
        assert victim is target  # Should hit target, not countermeasure

    def test_expired_countermeasure_ignored(self):
        """Test that expired countermeasures are ignored."""
        calc = MissileHitCalculator()

        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 45.0, -123.0, 1000.0)

        # Create expired countermeasure
        expired_cm = MockUnit(3, 45.0, -123.0, 1000.0)
        expired_cm.is_countermeasure = True
        expired_cm.is_destroyed = True  # Marked as destroyed/expired

        calc.begin_tick_snapshot([missile, target, expired_cm])

        # Should ignore expired countermeasure
        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [expired_cm], 1.0)

        assert hit is False  # Expired countermeasure is ignored
        assert victim is None

    def test_active_countermeasure_in_potential_targets(self):
        """Test that active countermeasures in target list don't cause hits."""
        calc = MissileHitCalculator(HitParams(fuse_radius_m=200.0))

        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 45.001, -123.0, 1000.0)  # ~111m away, within 200m radius

        # Active countermeasure at same position as target
        active_cm = MockUnit(3, 45.001, -123.0, 1000.0)
        active_cm.is_countermeasure = True
        active_cm.is_missile = False
        active_cm.is_destroyed = False

        calc.begin_tick_snapshot([missile, target, active_cm])

        # Even if countermeasure is in potential targets, it should not be hit
        # The missile should only target its designated target
        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target], 1.0)

        assert hit is True
        assert victim is target  # Should hit actual target


class TestMissileHitCalculatorIntegration:
    """Integration tests for complete hit calculation scenarios."""

    def test_continuous_collision_detection_scenario(self):
        """Test complete CCD scenario with moving objects."""
        calc = MissileHitCalculator(HitParams(fuse_radius_m=100.0))

        # Create missile and target at t0
        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 45.001, -123.0, 1000.0)  # ~111m north

        # Take t0 snapshot
        calc.begin_tick_snapshot([missile, target])

        # Move objects (simulate t1 positions after movement)
        missile.position.lat = 45.0005  # Moved north 55m
        target.position.lat = 45.0005  # Also moved north 55m

        # They should be close enough to hit during the tick
        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target], 1.0)

        assert hit is True
        assert victim is target
        assert 0.0 <= t_frac <= 1.0

    def test_near_miss_scenario(self):
        """Test scenario where objects pass close but don't hit."""
        calc = MissileHitCalculator(HitParams(fuse_radius_m=10.0))  # Small radius

        # Objects start apart and move in parallel
        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 45.0001, -123.0, 1000.0)  # 11m north

        calc.begin_tick_snapshot([missile, target])

        # Both move east in parallel (perpendicular to separation)
        missile.position.lon = -122.999
        target.position.lon = -122.999

        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target], 1.0)

        assert hit is False  # Too far apart for small fuse radius

    def test_overtaking_scenario(self):
        """Test scenario where missile overtakes target."""
        calc = MissileHitCalculator(HitParams(fuse_radius_m=50.0))

        # Missile behind target initially
        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 45.001, -123.0, 1000.0)  # Target ahead

        calc.begin_tick_snapshot([missile, target])

        # Missile moves faster and overtakes
        missile.position.lat = 45.002  # Missile moves ahead
        target.position.lat = 45.0015  # Target moves slower

        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target], 1.0)

        assert hit is True  # Should hit during overtaking
        assert 0.0 < t_frac < 1.0  # Hit occurs during tick, not at endpoints

    def test_multiple_targets_first_hit(self):
        """Test with multiple targets where first valid hit is returned."""
        calc = MissileHitCalculator(HitParams(fuse_radius_m=100.0))  # Smaller radius

        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target1 = MockUnit(2, 45.002, -123.0, 1000.0)  # Far (~222m, outside 100m radius)
        target2 = MockUnit(3, 45.0002, -123.0, 1000.0)  # Close (~22m, inside)
        target3 = MockUnit(4, 45.0001, -123.0, 1000.0)  # Closer (~11m, inside)

        calc.begin_tick_snapshot([missile, target1, target2, target3])

        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target1, target2, target3], 1.0)

        # Should hit the first valid target (target2, since target1 is too far)
        assert hit is True
        assert victim is target2

    def test_3d_collision_with_altitude_difference(self):
        """Test collision detection with altitude differences."""
        calc = MissileHitCalculator(HitParams(fuse_radius_m=200.0))

        # Objects at different altitudes
        missile = MockUnit(1, 45.0, -123.0, 1000.0)
        target = MockUnit(2, 45.0, -123.0, 1150.0)  # 150m higher

        calc.begin_tick_snapshot([missile, target])

        # Both move but maintain relative positions
        missile.position.lon = -122.999
        target.position.lon = -122.999

        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target], 1.0)

        assert hit is True  # 150m < 200m fuse radius

    def test_realistic_missile_intercept(self):
        """Test realistic missile intercept scenario."""
        calc = MissileHitCalculator(HitParams(fuse_radius_m=300.0))

        # Missile approaching target from behind and below
        missile = MockUnit(1, 44.995, -123.005, 800.0)  # Behind and below
        target = MockUnit(2, 45.0, -123.0, 1000.0)  # Target

        calc.begin_tick_snapshot([missile, target])

        # Missile catches up during tick
        missile.position.lat = 45.001  # Missile passes target
        missile.position.lon = -122.999
        missile.position.alt = 1050.0

        # Target continues forward
        target.position.lat = 45.001
        target.position.lon = -122.998

        hit, victim, t_frac = calc.check_and_maybe_hit(missile, [target], 1.0)

        assert hit is True
        assert victim is target
        assert 0.0 <= t_frac <= 1.0
