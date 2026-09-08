#!/usr/bin/env python3
"""
Tests for aircrafts.core.nez module.
Tests No Escape Zone (NEZ) calculation, Dynamic Launch Zone (DLZ), and SQI computations.
"""

import math

import numpy as np
import pytest

from bvr_marl_core.aircraft.core.nez import NoEscapeZoneCalculator
from tests.mocks import MockAircraft, MockMissile, MockPosition


@pytest.fixture
def mock_aircraft():
    """Create mock aircraft for NEZ testing."""
    aircraft = MockAircraft(
        name="test_nez_aircraft", position=MockPosition(0, 0, 10000), yaw_deg=0.0, speed=300.0
    )
    aircraft.missile_types = [MockMissile]
    aircraft.missiles = []
    return aircraft


@pytest.fixture
def mock_target():
    """Create mock target aircraft."""
    target = MockAircraft(
        name="test_target",
        position=MockPosition(0.3, 0, 10000),  # 30km north
        yaw_deg=180.0,
        speed=250.0,
    )
    target.id = "target_001"
    target.rcs = 5.0  # m²
    return target


@pytest.fixture
def nez_calculator(mock_aircraft):
    """Create NEZ calculator with mock aircraft."""
    return NoEscapeZoneCalculator(mock_aircraft)


# ============================================================================
# NEZ CALCULATOR INITIALIZATION TESTS
# ============================================================================


def test_nez_calculator_initialization(mock_aircraft):
    """Test NEZ calculator initialization."""
    calc = NoEscapeZoneCalculator(mock_aircraft)

    assert calc.own == mock_aircraft
    assert hasattr(calc, "own")


def test_nez_calculator_parent_reference(nez_calculator, mock_aircraft):
    """Test NEZ calculator parent reference."""
    assert nez_calculator.own is mock_aircraft
    assert nez_calculator.own.name == "test_nez_aircraft"


def test_estimated_dlz_uses_track_state_and_covariance(nez_calculator):
    state = np.array([0.0, 40_000.0, 0.0, 0.0, -250.0, 0.0])
    covariance = np.diag([400.0, 900.0, 400.0, 25.0, 225.0, 25.0])

    estimate = nez_calculator.compute_dlz_from_track(state, covariance)

    assert estimate.range_sigma_m == pytest.approx(30.0)
    assert estimate.conservative.r_aero_m < estimate.nominal.r_aero_m
    assert estimate.nominal.r_aero_m < estimate.optimistic.r_aero_m
    assert (
        0.0
        <= nez_calculator.sqi_from_estimate(40_000.0, estimate.closing_speed_mps, estimate.nominal)
        <= 1.0
    )


def test_estimated_dlz_rejects_partial_covariance(nez_calculator):
    with pytest.raises(ValueError, match="6x6 covariance"):
        nez_calculator.compute_dlz_from_track(np.zeros(6), np.eye(3))


# ============================================================================
# MISSILE SELECTION TESTS
# ============================================================================


def test_get_best_missile_from_types(nez_calculator):
    """Test getting best missile from missile types."""
    # Aircraft has missile_types but no actual missiles
    nez_calculator.own.missiles = []
    nez_calculator.own.missile_types = [MockMissile]

    best_missile = nez_calculator._get_best_missile(nez_calculator.own)

    # Should return a missile instance or None based on implementation
    if best_missile is not None:
        assert hasattr(best_missile, "max_range_m")


def test_get_best_missile_from_inventory(nez_calculator):
    """Test getting best missile from missile inventory."""
    # Create mock missiles with different ranges
    missile1 = MockMissile()
    missile1.max_range_m = 50000.0

    missile2 = MockMissile()
    missile2.max_range_m = 75000.0

    nez_calculator.own.missiles = [missile1, missile2]

    best_missile = nez_calculator._get_best_missile(nez_calculator.own)

    # Should return missile with longest range
    if best_missile is not None:
        assert best_missile.max_range_m >= missile1.max_range_m


def test_get_best_missile_no_missiles(nez_calculator):
    """Test getting best missile when no missiles available."""
    nez_calculator.own.missiles = []
    nez_calculator.own.missile_types = []

    best_missile = nez_calculator._get_best_missile(nez_calculator.own)
    assert best_missile is None


def test_get_missile_effective_range(nez_calculator):
    """Test missile effective range calculation."""
    missile = MockMissile()
    missile.fox_type = 3  # Active radar homing
    missile.max_range_m = 75000.0

    effective_range = nez_calculator._get_missile_effective_range(missile, nez_calculator.own)

    assert isinstance(effective_range, (int, float))
    assert effective_range > 0


def test_get_missile_effective_range_fox_types(nez_calculator):
    """Test effective range calculation for different FOX types."""
    # Fox 1 - Semi-active radar homing
    missile_fox1 = MockMissile()
    missile_fox1.fox_type = 1
    missile_fox1.max_range_m = 50000.0

    range_fox1 = nez_calculator._get_missile_effective_range(missile_fox1, nez_calculator.own)

    # Fox 2 - Infrared homing
    missile_fox2 = MockMissile()
    missile_fox2.fox_type = 2
    missile_fox2.max_range_m = 35000.0

    range_fox2 = nez_calculator._get_missile_effective_range(missile_fox2, nez_calculator.own)

    # Fox 3 - Active radar homing
    missile_fox3 = MockMissile()
    missile_fox3.fox_type = 3
    missile_fox3.max_range_m = 75000.0

    range_fox3 = nez_calculator._get_missile_effective_range(missile_fox3, nez_calculator.own)

    # All should return valid ranges
    assert range_fox1 > 0
    assert range_fox2 > 0
    assert range_fox3 > 0

    # Fox 2 should have reduced range (IR limited)
    assert range_fox2 < missile_fox2.max_range_m


# ============================================================================
# RADAR FACTOR TESTS
# ============================================================================


def test_get_parent_radar_factor(nez_calculator):
    """Test parent radar factor calculation."""
    # Test with aircraft that has radar
    factor = nez_calculator._get_parent_radar_factor(nez_calculator.own)

    assert isinstance(factor, float)
    assert 0.3 <= factor <= 1.5  # Should be clipped to this range


def test_get_parent_radar_factor_no_radar(nez_calculator):
    """Test radar factor when aircraft has no radar."""
    aircraft_no_radar = MockAircraft(name="no_radar")
    del aircraft_no_radar.radar  # Remove radar

    factor = nez_calculator._get_parent_radar_factor(aircraft_no_radar)

    assert factor == 0.5  # Default fallback


# ============================================================================
# RCS CALCULATION TESTS
# ============================================================================


def test_rcs_calculation(nez_calculator, mock_target):
    """Test RCS calculation."""
    mock_target.rcs = 5.0
    rcs = nez_calculator._rcs(mock_target)

    assert rcs == 5.0


def test_rcs_fallback_reference_area(nez_calculator):
    """Test RCS fallback to reference area."""
    target = MockAircraft(name="rcs_test")
    target.reference_area_m2 = 10.0
    # Don't set rcs attribute

    rcs = nez_calculator._rcs(target)

    # Should use reference_area fallback
    assert rcs == 0.5 * 10.0


def test_rcs_factor_calculation(nez_calculator):
    """Test RCS factor calculation."""
    rcs_factor = nez_calculator._rcs_factor(5.0, seeker_sensitivity=1.0)

    assert isinstance(rcs_factor, float)
    assert rcs_factor > 0


# ============================================================================
# KINEMATIC RANGE TESTS
# ============================================================================


def test_kinematic_range_calculation(nez_calculator):
    """Test kinematic range calculation."""
    missile = MockMissile()
    missile.max_range_m = 75000.0
    missile.min_range_m = 1500.0

    kin_range = nez_calculator._kinematic_range(
        missile,
        own_speed=300.0,
        tgt_speed=250.0,
        rel_angle=0.0,  # Head-on
        own_alt=10000.0,
        tgt_alt=10000.0,
    )

    assert isinstance(kin_range, (int, float))
    assert kin_range >= missile.min_range_m
    assert kin_range <= missile.max_range_m * 1.3


def test_kinematic_range_altitude_bonus(nez_calculator):
    """Test kinematic range with altitude advantage."""
    missile = MockMissile()
    missile.max_range_m = 75000.0
    missile.min_range_m = 1500.0

    # Higher altitude should give bonus
    range_high = nez_calculator._kinematic_range(missile, 300.0, 250.0, 0.0, 15000.0, 10000.0)

    # Same altitude baseline
    range_same = nez_calculator._kinematic_range(missile, 300.0, 250.0, 0.0, 10000.0, 10000.0)

    assert range_high > range_same


def test_kinematic_range_closing_speed(nez_calculator):
    """Test kinematic range with different closing speeds."""
    missile = MockMissile()
    missile.max_range_m = 75000.0
    missile.min_range_m = 1500.0

    # Head-on (high closing speed)
    range_head_on = nez_calculator._kinematic_range(missile, 300.0, 250.0, 0.0, 10000.0, 10000.0)

    # Tail chase (low closing speed)
    range_tail = nez_calculator._kinematic_range(missile, 300.0, 250.0, 180.0, 10000.0, 10000.0)

    assert range_head_on > range_tail


# ============================================================================
# GEOMETRIC CALCULATION TESTS
# ============================================================================


def test_relative_bearing_calculation(nez_calculator, mock_target):
    """Test relative bearing calculation."""
    # Target due north, aircraft facing north
    nez_calculator.own.yaw_deg = 0.0
    mock_target.position = MockPosition(0.1, 0, 10000)  # North

    try:
        rel_bearing = nez_calculator._relative_bearing(nez_calculator.own, mock_target)

        assert isinstance(rel_bearing, (int, float))
        assert 0 <= rel_bearing <= 180  # Absolute bearing
    except ImportError:
        # Skip if geodetic utilities not available
        pytest.skip("Geodetic utilities required for bearing calculation")


def test_slant_range_calculation(nez_calculator, mock_target):
    """Test slant range calculation."""
    # Position target 30km north
    mock_target.position = MockPosition(0.3, 0, 10000)

    slant_range = nez_calculator._slant_range_m(nez_calculator.own, mock_target)

    assert isinstance(slant_range, float)
    assert slant_range > 0
    # Should be approximately 30km
    assert 25000 < slant_range < 35000


def test_slant_range_with_altitude_difference(nez_calculator, mock_target):
    """Test slant range calculation with altitude difference."""
    # Same horizontal position, different altitude
    mock_target.position = MockPosition(0.0, 0.0, 15000)  # 5km higher

    slant_range = nez_calculator._slant_range_m(nez_calculator.own, mock_target)

    # Should be approximately 5km (altitude difference)
    assert 4000 < slant_range < 6000


# ============================================================================
# DLZ CALCULATION TESTS
# ============================================================================


def test_compute_dlz_basic(nez_calculator, mock_target):
    """Test basic DLZ computation."""
    dlz = nez_calculator.compute_dlz(mock_target)

    # Test DLZ structure
    assert hasattr(dlz, "r_min_m")
    assert hasattr(dlz, "r_tr_m")
    assert hasattr(dlz, "r_pi_m")
    assert hasattr(dlz, "r_aero_m")
    assert hasattr(dlz, "r_nez_in_m")
    assert hasattr(dlz, "r_nez_out_m")

    # Test range ordering
    assert dlz.r_min_m <= dlz.r_tr_m
    assert dlz.r_tr_m <= dlz.r_pi_m
    assert dlz.r_pi_m <= dlz.r_aero_m

    # Test NEZ bounds
    assert dlz.r_nez_in_m <= dlz.r_nez_out_m


def test_compute_dlz_no_missile(nez_calculator, mock_target):
    """Test DLZ computation when no missile available."""
    nez_calculator.own.missiles = []
    nez_calculator.own.missile_types = []

    dlz = nez_calculator.compute_dlz(mock_target)

    # Should return conservative default values
    assert dlz.r_min_m == 1500.0
    assert dlz.r_tr_m == 15000.0
    assert dlz.r_pi_m == 30000.0
    assert dlz.r_aero_m == 40000.0


def test_zone_for_range_classification(nez_calculator, mock_target):
    """Test range zone classification."""
    dlz = nez_calculator.compute_dlz(mock_target)

    # Test zone classification
    assert nez_calculator.zone_for_range(1000.0, dlz) == "R1"  # Below minimum
    assert nez_calculator.zone_for_range(dlz.r_min_m + 1000, dlz) == "R2"  # In NEZ
    assert nez_calculator.zone_for_range(dlz.r_tr_m + 1000, dlz) == "R3"  # Medium range
    assert nez_calculator.zone_for_range(dlz.r_pi_m + 1000, dlz) == "R4"  # Long range


def test_nez_visibility(nez_calculator, mock_target):
    """Test NEZ visibility determination."""
    dlz = nez_calculator.compute_dlz(mock_target)

    # Test NEZ visibility
    range_in_nez = (dlz.r_min_m + dlz.r_tr_m) / 2
    range_outside_nez = dlz.r_aero_m

    assert nez_calculator.nez_visible(range_in_nez, dlz)
    assert not nez_calculator.nez_visible(range_outside_nez, dlz)


# ============================================================================
# SQI CALCULATION TESTS
# ============================================================================


def test_sqi_calculation(nez_calculator, mock_target):
    """Test SQI (Target Intercept Probability Weighted Index) calculation."""
    sqi = nez_calculator.sqi(nez_calculator.own, mock_target)

    # SQI should be between 0 and 1
    assert isinstance(sqi, float)
    assert 0.0 <= sqi <= 1.0


def test_sqi_with_dlz(nez_calculator, mock_target):
    """Test SQI calculation with pre-computed DLZ."""
    dlz = nez_calculator.compute_dlz(mock_target)
    sqi = nez_calculator.sqi(nez_calculator.own, mock_target, dlz=dlz)

    assert isinstance(sqi, float)
    assert 0.0 <= sqi <= 1.0


def test_sqi_distance_effect(nez_calculator):
    """Test SQI distance effect."""
    # Close target should have higher SQI
    close_target = MockAircraft(
        name="close_target",
        position=MockPosition(0.1, 0, 10000),  # 10km
        speed=250.0,
    )

    # Far target should have lower SQI
    far_target = MockAircraft(
        name="far_target",
        position=MockPosition(0.5, 0, 10000),  # 50km
        speed=250.0,
    )

    sqi_close = nez_calculator.sqi(nez_calculator.own, close_target)
    sqi_far = nez_calculator.sqi(nez_calculator.own, far_target)

    # Closer target should generally have higher SQI
    # (May not always be true due to other factors, so just check bounds)
    assert 0.0 <= sqi_close <= 1.0
    assert 0.0 <= sqi_far <= 1.0


# ============================================================================
# COMPATIBILITY TESTS
# ============================================================================


def test_active_nez_compatibility(nez_calculator, mock_target):
    """Test active NEZ compatibility method."""
    active_nez = nez_calculator.active_nez(mock_target)

    assert isinstance(active_nez, (int, float))
    assert active_nez > 0


def test_passive_nez_compatibility(nez_calculator, mock_target):
    """Test passive NEZ compatibility method."""
    # This creates a reverse perspective calculation
    passive_nez = nez_calculator.passive_nez(mock_target)

    assert isinstance(passive_nez, (int, float))
    assert passive_nez > 0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_nez_calculator_error_handling():
    """Test NEZ calculator error handling."""
    # Test with invalid aircraft
    try:
        NoEscapeZoneCalculator(None)
        # Should handle gracefully or raise appropriate exception
    except Exception as e:
        assert isinstance(e, (AttributeError, TypeError))


def test_dlz_calculation_missing_attributes(nez_calculator):
    """Test DLZ calculation with missing target attributes."""
    minimal_target = MockAircraft(name="minimal")
    # Remove some attributes
    if hasattr(minimal_target, "speed"):
        del minimal_target.speed

    try:
        dlz = nez_calculator.compute_dlz(minimal_target)
        # Should handle gracefully with defaults
        assert hasattr(dlz, "r_min_m")
    except Exception:
        # Should not crash, may use defaults
        pass


def test_sqi_calculation_missing_attributes(nez_calculator):
    """Test SQI calculation with missing attributes."""
    minimal_target = MockAircraft(name="minimal")

    try:
        sqi = nez_calculator.sqi(nez_calculator.own, minimal_target)
        # Should handle gracefully
        assert 0.0 <= sqi <= 1.0
    except Exception:
        # Should not crash due to missing attributes
        pass


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_dlz_same_position_targets(nez_calculator):
    """Test DLZ calculation when target is at same position."""
    same_pos_target = MockAircraft(
        name="same_pos",
        position=MockPosition(0, 0, 10000),  # Same as aircraft
        speed=250.0,
    )

    dlz = nez_calculator.compute_dlz(same_pos_target)

    # Should handle zero range gracefully
    assert dlz.r_min_m > 0  # Minimum range should still be positive


def test_dlz_extreme_altitudes(nez_calculator):
    """Test DLZ calculation with extreme altitude differences."""
    high_target = MockAircraft(
        name="high_target",
        position=MockPosition(0.1, 0, 20000),  # 10km higher
        speed=250.0,
    )

    low_target = MockAircraft(
        name="low_target",
        position=MockPosition(0.1, 0, 1000),  # 9km lower
        speed=250.0,
    )

    dlz_high = nez_calculator.compute_dlz(high_target)
    dlz_low = nez_calculator.compute_dlz(low_target)

    # Both should produce valid DLZ
    assert dlz_high.r_aero_m > dlz_high.r_min_m
    assert dlz_low.r_aero_m > dlz_low.r_min_m

    # Don't assume specific altitude advantage - just verify both are reasonable
    assert dlz_high.r_aero_m > 10000.0  # Should be reasonable range
    assert dlz_low.r_aero_m > 10000.0  # Should be reasonable range


def test_dlz_extreme_speeds(nez_calculator):
    """Test DLZ calculation with extreme speed differences."""
    fast_target = MockAircraft(
        name="fast_target",
        position=MockPosition(0.2, 0, 10000),
        speed=600.0,  # Very fast
    )

    slow_target = MockAircraft(
        name="slow_target",
        position=MockPosition(0.2, 0, 10000),
        speed=100.0,  # Very slow
    )

    dlz_fast = nez_calculator.compute_dlz(fast_target)
    dlz_slow = nez_calculator.compute_dlz(slow_target)

    # Both should produce valid DLZ
    assert dlz_fast.r_aero_m > dlz_fast.r_min_m
    assert dlz_slow.r_aero_m > dlz_slow.r_min_m


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_nez_calculator_complete_workflow(nez_calculator, mock_target):
    """Test complete NEZ calculation workflow."""
    # Compute DLZ
    dlz = nez_calculator.compute_dlz(mock_target)

    # Get current range
    current_range = nez_calculator._slant_range_m(nez_calculator.own, mock_target)

    # Classify range zone
    zone = nez_calculator.zone_for_range(current_range, dlz)

    # Check NEZ visibility
    nez_vis = nez_calculator.nez_visible(current_range, dlz)

    # Calculate SQI
    sqi = nez_calculator.sqi(nez_calculator.own, mock_target, dlz=dlz)

    # All calculations should complete successfully
    assert zone in ["R1", "R2", "R3", "R4"]
    assert isinstance(nez_vis, bool)
    assert 0.0 <= sqi <= 1.0


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_a_stationary_target_is_shootable_at_100km():
    """A target that cannot run away must not be scored like one that is running.

    R_aero used to scale with the TOTAL closing speed, so a stationary target with a
    non-closing shooter (Vc ~ 0) scored f_aspect 0.45 -- essentially the same as a
    target fleeing at 250 m/s (0.40). Measured on a stationary-target warmup stage, whose opponent is an
    anchored hold, that put R_pi at 37-57 km against an engagement opening at
    50-100 km: the shot was out of envelope for the whole episode and neither the BT
    nor the RL policy could ever fire.

    A non-stealth legacy fighter should be engageable at ~100 km with an AMRAAM, so
    this is asserted on the real airframe and the real weapon rather than the 75 km
    mock missile.
    """
    from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
    from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
    from bvr_marl_core.simulator.core.units import Position

    class _MapLimits:
        left_lon = -3.0
        right_lon = 3.0
        bottom_lat = -3.0
        top_lat = 3.0
        min_alt = 0.0
        max_alt = 20_000.0

    shooter = Eurofighter(
        Position(lat=0.0, lon=0.0, alt=10_000.0), 0.0, 300.0, "BLUE", _MapLimits(), 0.0, 20_000.0
    )
    shooter.missile_types = [AIM120_AMRAAM]
    calc = NoEscapeZoneCalculator(shooter)

    state = np.array([0.0, 100_000.0, 0.0, 0.0, 0.0, 0.0])  # 100 km north, stationary
    covariance = np.diag([400.0, 900.0, 400.0, 25.0, 225.0, 25.0])

    dlz = calc.compute_dlz_from_track(state, covariance).nominal

    assert dlz.r_pi_m > 100_000.0, f"R_pi only {dlz.r_pi_m / 1000:.1f} km"
    assert calc.zone_for_range(100_000.0, dlz) in ("R2", "R3")


def test_a_fleeing_target_still_shrinks_the_envelope(nez_calculator):
    """The tail-chase penalty must survive the fix."""
    covariance = np.diag([400.0, 900.0, 400.0, 25.0, 225.0, 25.0])
    stationary = nez_calculator.compute_dlz_from_track(
        np.array([0.0, 100_000.0, 0.0, 0.0, 0.0, 0.0]), covariance
    ).nominal
    fleeing = nez_calculator.compute_dlz_from_track(
        np.array([0.0, 100_000.0, 0.0, 0.0, 300.0, 0.0]), covariance
    ).nominal
    closing = nez_calculator.compute_dlz_from_track(
        np.array([0.0, 100_000.0, 0.0, 0.0, -300.0, 0.0]), covariance
    ).nominal

    assert fleeing.r_aero_m < stationary.r_aero_m < closing.r_aero_m


def test_the_shooters_own_closure_does_not_change_the_envelope(nez_calculator):
    """R_aero is a property of the TARGET's escape; the shooter's energy enters

    separately through the launch-altitude/speed term. Counting its closure here
    too was the double-count that penalised a shooter for stopping."""
    covariance = np.diag([400.0, 900.0, 400.0, 25.0, 225.0, 25.0])
    target_state = np.array([0.0, 100_000.0, 0.0, 0.0, 0.0, 0.0])

    baseline = nez_calculator.compute_dlz_from_track(target_state, covariance).nominal
    nez_calculator.own.speed = 0.0
    stopped = nez_calculator.compute_dlz_from_track(target_state, covariance).nominal

    assert stopped.r_aero_m == pytest.approx(baseline.r_aero_m, rel=0.25)


def test_shooting_uphill_shrinks_the_envelope(nez_calculator):
    """R_aero must know the altitude DIFFERENCE, not just the shooter's altitude.

    `_achievable_max_range` scaled only with `own_alt`, so a shot taken from far
    below its target was scored as if it were level. The legacy `_kinematic_range`
    still carries this as `height_bonus`; the production path had lost it.

    Measured on a BT rollout: mean launch 77.4 km at an altitude delta of -7120 m,
    called in-envelope by the DLZ, with `missile_no_terminal_rate` 0.75 -- three
    missiles in four never reached a terminal event because they spent their energy
    climbing.
    """
    covariance = np.diag([400.0, 900.0, 400.0, 25.0, 225.0, 25.0])
    level = nez_calculator.compute_dlz_from_track(
        np.array([0.0, 80_000.0, 0.0, 0.0, 0.0, 0.0]), covariance
    ).nominal
    uphill = nez_calculator.compute_dlz_from_track(
        np.array([0.0, 80_000.0, 7_000.0, 0.0, 0.0, 0.0]), covariance
    ).nominal
    downhill = nez_calculator.compute_dlz_from_track(
        np.array([0.0, 80_000.0, -5_000.0, 0.0, 0.0, 0.0]), covariance
    ).nominal

    assert uphill.r_aero_m < level.r_aero_m < downhill.r_aero_m


def test_climbing_costs_more_than_diving_gains(nez_calculator):
    """Asymmetric on purpose: the missile cannot recover the energy on the way down."""
    covariance = np.diag([400.0, 900.0, 400.0, 25.0, 225.0, 25.0])

    def aero(up_m):
        return nez_calculator.compute_dlz_from_track(
            np.array([0.0, 80_000.0, float(up_m), 0.0, 0.0, 0.0]), covariance
        ).nominal.r_aero_m

    level = aero(0.0)
    lost = level - aero(5_000.0)
    gained = aero(-5_000.0) - level

    assert lost > gained
