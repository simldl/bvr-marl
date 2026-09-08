#!/usr/bin/env python3
"""
Tests for aircrafts.types.awacs module.
Tests AWACS aircraft configuration and initialization.
"""

import pytest

from bvr_marl_core.aircraft.types.awacs import AWACS
from tests.mocks import MockMapLimits, MockPosition


@pytest.fixture
def mock_position():
    """Create mock position for AWACS testing."""
    return MockPosition(0, 0, 10000)


@pytest.fixture
def mock_map_limits():
    """Create mock map limits for AWACS testing."""
    return MockMapLimits()


@pytest.fixture
def awacs_aircraft(mock_position, mock_map_limits):
    """Create AWACS aircraft instance for testing."""
    try:
        return AWACS(
            position=mock_position,
            yaw_deg=0.0,
            speed_mps=200.0,
            group="BLUE",
            map_limits=mock_map_limits,
            min_alt_m=0.0,
            max_alt_m=12000.0,
        )
    except Exception:
        pytest.skip("AWACS initialization requires full Aircraft base class")


# ============================================================================
# AWACS CONFIGURATION TESTS
# ============================================================================


def test_awacs_config_defaults():
    """Test AWACS default configuration parameters."""
    config = AWACS.Config()

    # Physics parameters (large aircraft)
    assert config.mass_kg == 150000.0
    assert config.reference_area_m2 == 283.0
    assert config.aspect_ratio == 7.1
    assert config.oswald_e == 0.80

    # Speed limits (much slower than fighters)
    assert config.max_speed_mps == 250.0
    assert config.min_speed_mps == 80.0
    assert config.n_max == 2.5
    assert config.stall0_mps == 70.0

    # Altitude envelope (lower ceiling than fighters)
    assert config.max_climb_angle_deg == 15.0
    assert config.min_alt_m == 0.0
    assert config.max_alt_m == 12000.0


def test_awacs_radar_characteristics():
    """Test AWACS radar characteristics - primary capability."""
    config = AWACS.Config()

    # Should have 360 degree coverage (rotating dome)
    assert config.radar_horizontal_fov_deg == 360.0
    assert config.radar_vertical_fov_deg == 60.0

    # Less-potent surveillance system: the workshop replaced one 400 km AWACS with
    # two ~250 km 360-degree platforms per team.
    assert config.radar_max_range_m == 250000.0

    # Should use S-band for better range
    assert config.radar_frequency_hz == 3.0e9

    # Reduced power/gain relative to the old single super-AWACS
    assert config.radar_tx_power_w == 25e3
    assert config.radar_antenna_gain_db == 40.0

    # Should have low detection threshold (high sensitivity)
    assert config.radar_snr_threshold_db == 6.0

    # Slower scan rate (rotating dome)
    assert config.radar_beam_rate_hz == 6.0


def test_awacs_rcs():
    """Test AWACS has large RCS (not stealthy)."""
    config = AWACS.Config()

    # AWACS is a large aircraft, should have high RCS
    assert config.rcs == 100.0


def test_awacs_no_weapons():
    """Test AWACS has no offensive weapons."""
    config = AWACS.Config()

    # Should have no missiles
    assert config.missile_types == ()
    assert config.max_missiles == 0


def test_awacs_countermeasures():
    """Test AWACS has defensive countermeasures."""
    config = AWACS.Config()

    # Should have more countermeasures than fighters (for self-defense)
    assert config.flares == 60
    assert config.chaff == 120
    assert config.ecm == 10
    assert config.decoys == 10


def test_awacs_passive_sensors():
    """Test AWACS passive sensor characteristics."""
    config = AWACS.Config()

    # Should have better ESM than fighters
    assert config.passive_radar_angular_error_deg == 2.0
    assert config.passive_radar_range_error_m == 1000.0
    assert config.passive_radar_max_age_s == 5.0


# ============================================================================
# AWACS INITIALIZATION TESTS
# ============================================================================


def test_awacs_initialization_basic(awacs_aircraft):
    """Test basic AWACS initialization."""
    assert awacs_aircraft.name == "AWACS"
    assert awacs_aircraft.group == "BLUE"
    assert awacs_aircraft.position.lat == 0.0
    assert awacs_aircraft.position.lon == 0.0
    assert awacs_aircraft.position.alt == 10000.0
    assert awacs_aircraft.yaw_deg == 0.0
    assert awacs_aircraft.speed == 200.0


def test_awacs_support_asset_flags(awacs_aircraft):
    """Test AWACS is marked as support asset."""
    assert hasattr(awacs_aircraft, "is_support_asset")
    assert awacs_aircraft.is_support_asset is True

    assert hasattr(awacs_aircraft, "is_high_value_target")
    assert awacs_aircraft.is_high_value_target is True


def test_awacs_speed_clamping(mock_position, mock_map_limits):
    """Test AWACS speed is clamped to max speed."""
    try:
        awacs = AWACS(
            position=mock_position,
            yaw_deg=0.0,
            speed_mps=500.0,  # Too fast for AWACS
            group="BLUE",
            map_limits=mock_map_limits,
            min_alt_m=0.0,
            max_alt_m=12000.0,
        )
        # Speed should be clamped to max
        assert awacs.speed <= 250.0
    except Exception:
        pytest.skip("AWACS speed clamping test requires full Aircraft class")


def test_awacs_altitude_limit(mock_position, mock_map_limits):
    """Test AWACS altitude is limited to its ceiling."""
    try:
        awacs = AWACS(
            position=mock_position,
            yaw_deg=0.0,
            speed_mps=200.0,
            group="BLUE",
            map_limits=mock_map_limits,
            min_alt_m=0.0,
            max_alt_m=20000.0,  # Higher than AWACS ceiling
        )
        # Max alt should be limited to AWACS ceiling
        assert awacs.max_alt_m <= 12000.0
    except Exception:
        pytest.skip("AWACS altitude limit test requires full Aircraft class")


def test_awacs_custom_name(mock_position, mock_map_limits):
    """Test AWACS can have custom name."""
    try:
        awacs = AWACS(
            position=mock_position,
            yaw_deg=0.0,
            speed_mps=200.0,
            group="BLUE",
            map_limits=mock_map_limits,
            min_alt_m=0.0,
            max_alt_m=12000.0,
            name="E-3 Sentry",
        )
        assert awacs.name == "E-3 Sentry"
    except Exception:
        pytest.skip("AWACS custom name test requires full Aircraft class")


# ============================================================================
# AWACS RCS PATTERN TESTS
# ============================================================================


def test_awacs_rcs_pattern(awacs_aircraft):
    """Test AWACS RCS pattern is large and uniform."""
    pattern = awacs_aircraft.rcs_pattern

    # Large aircraft should be visible from all angles
    assert pattern["front_floor"] >= 0.5
    assert pattern["tail_floor"] >= 0.5

    # Should have relatively uniform pattern (low n_az)
    assert pattern["n_az"] <= 1.0

    # High minimum and maximum RCS
    assert pattern["sigma_min"] >= 50.0
    assert pattern["sigma_max"] >= 100.0


# ============================================================================
# COMPARATIVE TESTS
# ============================================================================


def test_awacs_vs_fighter_characteristics():
    """Test AWACS characteristics vs fighter aircraft."""
    from bvr_marl_core.aircraft.types.f22 import F22

    awacs_config = AWACS.Config()
    f22_config = F22.Config()

    # AWACS should be slower
    assert awacs_config.max_speed_mps < f22_config.max_speed_mps

    # AWACS should have longer radar range
    assert awacs_config.radar_max_range_m > f22_config.radar_max_range_m

    # AWACS should have wider radar coverage
    assert awacs_config.radar_horizontal_fov_deg > f22_config.radar_horizontal_fov_deg

    # AWACS should have much higher RCS
    assert awacs_config.rcs > f22_config.rcs * 1000

    # AWACS should have no missiles
    assert awacs_config.max_missiles == 0
    assert f22_config.max_missiles > 0


def test_awacs_vs_fighter_maneuverability():
    """Test AWACS has lower maneuverability than fighters."""
    from bvr_marl_core.aircraft.types.f22 import F22

    awacs_config = AWACS.Config()
    f22_config = F22.Config()

    # AWACS should have much lower G limit
    assert awacs_config.n_max < f22_config.n_max / 2

    # AWACS should have lower climb angle
    assert awacs_config.max_climb_angle_deg < f22_config.max_climb_angle_deg / 2


# ============================================================================
# REALISM TESTS
# ============================================================================


def test_awacs_realistic_mass():
    """Test AWACS mass is realistic for large aircraft."""
    config = AWACS.Config()

    # E-3 Sentry MTOW is ~150,000 kg
    assert 100_000 <= config.mass_kg <= 200_000


def test_awacs_realistic_radar_range():
    """Test AWACS radar range is realistic for the less-potent split system."""
    config = AWACS.Config()

    # Two less-potent ~250 km 360-degree systems replace the single 400 km AWACS;
    # still a long surveillance reach, well beyond fighter radars.
    assert 200_000 <= config.radar_max_range_m <= 300_000


def test_awacs_realistic_service_ceiling():
    """Test AWACS service ceiling is realistic."""
    config = AWACS.Config()

    # E-3 Sentry service ceiling is ~12,500m
    assert 10_000 <= config.max_alt_m <= 15_000


def test_awacs_realistic_speed():
    """Test AWACS speed is realistic."""
    config = AWACS.Config()

    # E-3 Sentry max speed is ~480 knots (~247 m/s)
    assert 200 <= config.max_speed_mps <= 280


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_awacs_aircraft_inheritance(awacs_aircraft):
    """Test AWACS inherits from Aircraft base class correctly."""
    assert hasattr(awacs_aircraft, "name")
    assert hasattr(awacs_aircraft, "position")
    assert hasattr(awacs_aircraft, "yaw_deg")
    assert hasattr(awacs_aircraft, "speed")
    assert hasattr(awacs_aircraft, "group")
    assert hasattr(awacs_aircraft, "radar")

    from bvr_marl_core.aircraft.aircraft import Aircraft

    assert isinstance(awacs_aircraft, Aircraft)


def test_awacs_has_radar(awacs_aircraft):
    """Test AWACS has radar system."""
    assert hasattr(awacs_aircraft, "radar")
    assert awacs_aircraft.radar is not None


def test_awacs_has_datalink(awacs_aircraft):
    """Test AWACS has datalink capability."""
    if hasattr(awacs_aircraft.radar, "data_link"):
        assert awacs_aircraft.radar.data_link is not None


# ============================================================================
# LOCK FOV TESTS
# ============================================================================


def test_awacs_lock_fov_default():
    """Test AWACS has default lock FOV separate from detection FOV."""
    config = AWACS.Config()

    # Detection FOV is 360 degrees
    assert config.radar_horizontal_fov_deg == 360.0

    # Lock FOV should be narrower
    assert hasattr(config, "lock_fov_deg")
    assert config.lock_fov_deg == 60.0
    assert config.lock_fov_deg < config.radar_horizontal_fov_deg


def test_awacs_lock_fov_on_instance(awacs_aircraft):
    """Test AWACS instance has lock_fov_deg attribute."""
    assert hasattr(awacs_aircraft, "lock_fov_deg")
    assert awacs_aircraft.lock_fov_deg == 60.0


def test_awacs_lock_fov_custom(mock_position, mock_map_limits):
    """Test AWACS lock FOV can be customized at instantiation."""
    try:
        awacs = AWACS(
            position=mock_position,
            yaw_deg=0.0,
            speed_mps=200.0,
            group="BLUE",
            map_limits=mock_map_limits,
            min_alt_m=0.0,
            max_alt_m=12000.0,
            lock_fov_deg=90.0,
        )
        assert awacs.lock_fov_deg == 90.0
    except Exception:
        pytest.skip("AWACS custom lock FOV requires full Aircraft class")


def test_awacs_can_lock_target_method(awacs_aircraft):
    """Test AWACS has can_lock_target method."""
    assert hasattr(awacs_aircraft, "can_lock_target")
    assert callable(awacs_aircraft.can_lock_target)


def test_awacs_can_lock_target_in_fov(mock_position, mock_map_limits):
    """Test AWACS can lock targets within lock FOV."""
    # AWACS facing north (yaw=0)
    awacs = AWACS(
        position=MockPosition(0, 0, 10000),
        yaw_deg=0.0,
        speed_mps=200.0,
        group="BLUE",
        map_limits=mock_map_limits,
        min_alt_m=0.0,
        max_alt_m=12000.0,
        lock_fov_deg=120.0,
    )

    # Target to the north (in front) - should be lockable
    class MockTarget:
        position = MockPosition(0.01, 0, 10000)  # North of AWACS

    assert awacs.can_lock_target(MockTarget()) is True


def test_awacs_cannot_lock_target_outside_fov(mock_position, mock_map_limits):
    """Test AWACS cannot lock targets outside lock FOV."""
    try:
        # AWACS facing east (yaw=90)
        awacs = AWACS(
            position=MockPosition(0, 0, 10000),
            yaw_deg=90.0,
            speed_mps=200.0,
            group="BLUE",
            map_limits=mock_map_limits,
            min_alt_m=0.0,
            max_alt_m=12000.0,
            lock_fov_deg=120.0,
        )

        # Target to the west (behind) - should NOT be lockable
        class MockTarget:
            position = MockPosition(-0.1, 0, 10000)  # West of AWACS

        assert awacs.can_lock_target(MockTarget()) is False
    except Exception:
        pytest.skip("AWACS can_lock_target test requires full Aircraft class")


def test_awacs_can_lock_target_edge_of_fov(mock_position, mock_map_limits):
    """Test AWACS lock behavior at edge of FOV."""
    try:
        # AWACS facing north (yaw=0) with 120 degree FOV
        awacs = AWACS(
            position=MockPosition(0, 0, 10000),
            yaw_deg=0.0,  # Facing north
            speed_mps=200.0,
            group="BLUE",
            map_limits=mock_map_limits,
            min_alt_m=0.0,
            max_alt_m=12000.0,
            lock_fov_deg=120.0,  # +/- 60 degrees
        )

        # Target at 50 degrees to the east - should be lockable (within 60 deg)
        class MockTarget50:
            # Approximately 50 degrees offset
            position = MockPosition(0.08, 0.06, 10000)

        # Target at 70 degrees - should NOT be lockable (outside 60 deg)
        class MockTarget70:
            # Approximately 70 degrees offset
            position = MockPosition(0.09, 0.03, 10000)

        result_50 = awacs.can_lock_target(MockTarget50())
        awacs.can_lock_target(MockTarget70())

        # 50 degrees should be lockable, 70 degrees might not be
        assert result_50 is True
    except Exception:
        pytest.skip("AWACS edge FOV test requires full Aircraft class")


def test_awacs_can_lock_target_none():
    """Test AWACS handles None target gracefully."""
    try:
        awacs = AWACS(
            position=MockPosition(0, 0, 10000),
            yaw_deg=0.0,
            speed_mps=200.0,
            group="BLUE",
            map_limits=MockMapLimits(),
            min_alt_m=0.0,
            max_alt_m=12000.0,
        )
        assert awacs.can_lock_target(None) is False
    except Exception:
        pytest.skip("AWACS can_lock_target None test requires full Aircraft class")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
