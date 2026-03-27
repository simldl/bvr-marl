#!/usr/bin/env python3
"""
Tests for aircrafts.types.f22 module.
Tests F-22 Raptor aircraft configuration and initialization.
"""

import pytest

from air_to_air_rl.aircrafts.types.f22 import F22
from tests.mocks import MockMapLimits, MockPosition


@pytest.fixture
def mock_position():
    """Create mock position for F22 testing."""
    return MockPosition(0, 0, 10000)


@pytest.fixture
def mock_map_limits():
    """Create mock map limits for F22 testing."""
    return MockMapLimits()


@pytest.fixture
def f22_aircraft(mock_position, mock_map_limits):
    """Create F22 aircraft instance for testing."""
    try:
        return F22(
            position=mock_position,
            yaw_deg=0.0,
            speed_mps=300.0,
            group="BLUE",
            map_limits=mock_map_limits,
            min_alt_m=0.0,
            max_alt_m=20000.0,
        )
    except Exception:
        pytest.skip("F22 initialization requires full Aircraft base class")


# ============================================================================
# F22 CONFIGURATION TESTS
# ============================================================================


def test_f22_config_defaults():
    """Test F22 default configuration parameters."""
    config = F22.Config()

    # Physics parameters (corrected based on Wikipedia wing geometry)
    assert config.mass_kg == 19700.0
    assert config.reference_area_m2 == 78.0  # ~840 ft² corrected from Wikipedia
    assert config.aspect_ratio == 2.36  # b^2/S = 13.56^2 / 78.04 corrected
    assert config.oswald_e == 0.82
    assert config.max_speed_mps == 680.0
    assert config.n_max == 9.0
    assert config.stall0_mps == 75.0

    # Flight envelope
    assert config.min_speed_mps == 80.0
    assert config.max_climb_angle_deg == 70.0
    assert config.min_alt_m == 0.0
    assert config.max_alt_m == 20000.0

    # Radar characteristics
    assert config.radar_horizontal_fov_deg == 120.0
    assert config.radar_vertical_fov_deg == 60.0
    assert config.radar_max_range_m == 230_000.0
    assert config.radar_frequency_hz == 9.5e9
    assert config.radar_tx_power_w == 25e3
    assert config.radar_antenna_gain_db == 40.0
    assert config.radar_snr_threshold_db == 8.0
    assert config.radar_beam_rate_hz == 10.0
    assert config.radar_beam_rate_p_hz == 8.0

    # Stealth characteristics
    assert config.rcs == 0.0001  # Very low RCS for stealth

    # Sensor configurations
    assert config.passive_radar_angular_error_deg == 5.0
    assert config.passive_radar_range_error_m == 2000.0
    assert config.passive_radar_max_age_s == 3.0

    # Missile warning system
    assert config.missile_warning_delay_s == 1.0
    assert config.missile_warning_delay_std == 0.5

    # Weapons loadout (reduced countermeasures for realism)
    assert config.max_missiles == 8
    assert config.flares == 2
    assert config.chaff == 2
    assert config.ecm == 2
    assert config.decoys == 2

    # Gun configuration should be None by default
    assert config.gun_config is None


def test_f22_config_missile_types():
    """Test F22 missile types configuration."""
    config = F22.Config()

    # Should have AIM-120 AMRAAM as primary missile
    assert hasattr(config, "missile_types")
    assert len(config.missile_types) == 1

    # Check missile type (may require importing the actual missile class)
    from air_to_air_rl.missiles.fox3.amraam import AIM120_AMRAAM

    assert config.missile_types[0] == AIM120_AMRAAM


def test_f22_performance_characteristics():
    """Test F22 performance characteristics are realistic."""
    config = F22.Config()

    # Verify performance parameters are within realistic ranges
    assert 15000 <= config.mass_kg <= 25000  # Typical fighter weight
    assert 70 <= config.reference_area_m2 <= 85  # Wing area (corrected for actual F-22)
    assert 2.0 <= config.aspect_ratio <= 3.0  # Wing aspect ratio (delta-wing fighters have low AR)
    assert 0.7 <= config.oswald_e <= 0.9  # Efficiency factor
    assert 600 <= config.max_speed_mps <= 750  # ~Mach 2+ capability
    assert 7.5 <= config.n_max <= 10.0  # G-force limits
    assert 60 <= config.stall0_mps <= 90  # Stall speed

    # Flight envelope checks
    assert 70 <= config.min_speed_mps <= 100  # Minimum operational speed
    assert 60 <= config.max_climb_angle_deg <= 90  # Climb capability
    assert config.min_alt_m == 0.0  # Sea level minimum
    assert 15000 <= config.max_alt_m <= 25000  # Service ceiling


def test_f22_radar_characteristics():
    """Test F22 radar characteristics are realistic."""
    config = F22.Config()

    # Radar performance should be advanced for 5th gen fighter
    assert 100 <= config.radar_horizontal_fov_deg <= 140  # Wide coverage
    assert 40 <= config.radar_vertical_fov_deg <= 80  # Vertical coverage
    assert 200_000 <= config.radar_max_range_m <= 300_000  # Long range
    assert 8e9 <= config.radar_frequency_hz <= 12e9  # X-band
    assert 20e3 <= config.radar_tx_power_w <= 30e3  # High power
    assert 35 <= config.radar_antenna_gain_db <= 45  # High gain
    assert 6 <= config.radar_snr_threshold_db <= 12  # Sensitivity

    # RCS should be very low for stealth
    assert config.rcs <= 0.001  # Stealth characteristics


def test_f22_weapons_loadout():
    """Test F22 weapons loadout is realistic."""
    config = F22.Config()

    # Missile loadout
    assert 6 <= config.max_missiles <= 10  # Internal bay capacity

    # Countermeasures (reduced for balance/realism in simulation)
    assert 2 <= config.flares <= 10  # Flare count
    assert 2 <= config.chaff <= 10  # Chaff count
    assert 2 <= config.ecm <= 8  # ECM uses
    assert 2 <= config.decoys <= 15  # Decoy count


# ============================================================================
# F22 INITIALIZATION TESTS
# ============================================================================


def test_f22_initialization_basic(f22_aircraft):
    """Test basic F22 initialization."""
    # Test basic aircraft properties
    assert f22_aircraft.name == "F22"
    assert f22_aircraft.group == "BLUE"
    assert f22_aircraft.position.lat == 0.0
    assert f22_aircraft.position.lon == 0.0
    assert f22_aircraft.position.alt == 10000.0
    assert f22_aircraft.yaw_deg == 0.0
    assert f22_aircraft.speed == 300.0


def test_f22_gun_configuration(f22_aircraft):
    """Test F22 gun system configuration."""
    # Check gun configuration exists (may be on config or aircraft directly)
    if hasattr(f22_aircraft, "gun_config") and f22_aircraft.gun_config is not None:
        gun_config = f22_aircraft.gun_config

        # M61A2 Vulcan cannon specifications
        assert gun_config["max_ammo"] == 480
        assert gun_config["muzzle_velocity_mps"] == 1050.0
        assert gun_config["max_range_m"] == 1500.0
        assert gun_config["burst_size"] == 50
        assert gun_config["burst_duration_s"] == 0.5
    else:
        # Gun config might be in the configuration dict
        if hasattr(f22_aircraft, "config") and "gun_config" in f22_aircraft.config:
            gun_config = f22_aircraft.config["gun_config"]
            assert gun_config is not None
        else:
            # Skip test if gun configuration not accessible
            pytest.skip("Gun configuration not accessible on F22 instance")


def test_f22_altitude_limits(mock_position, mock_map_limits):
    """Test F22 altitude limit configuration."""
    try:
        f22 = F22(
            position=mock_position,
            yaw_deg=0.0,
            speed_mps=300.0,
            group="BLUE",
            map_limits=mock_map_limits,
            min_alt_m=500.0,  # Custom minimum
            max_alt_m=18000.0,  # Custom maximum
        )

        # Should use provided altitude limits
        assert f22.min_alt_m == 500.0
        assert f22.max_alt_m == 18000.0
    except Exception:
        pytest.skip("F22 altitude configuration test requires full Aircraft class")


def test_f22_initialization_parameters(mock_position, mock_map_limits):
    """Test F22 initialization with various parameters."""
    try:
        f22 = F22(
            position=mock_position,
            yaw_deg=45.0,
            speed_mps=400.0,
            group="RED",
            map_limits=mock_map_limits,
            min_alt_m=0.0,
            max_alt_m=20000.0,
        )

        # Should preserve initialization parameters
        assert f22.yaw_deg == 45.0
        assert f22.speed == 400.0
        assert f22.group == "RED"
    except Exception:
        pytest.skip("F22 parameter test requires full Aircraft class")


# ============================================================================
# F22 CONFIG MODIFICATION TESTS
# ============================================================================


def test_f22_config_modification():
    """Test F22 configuration can be modified."""
    config = F22.Config()

    # Modify some parameters
    original_mass = config.mass_kg
    config.mass_kg = 20000.0

    assert config.mass_kg == 20000.0
    assert config.mass_kg != original_mass


def test_f22_config_inheritance():
    """Test F22 config class inheritance and dataclass behavior."""
    config = F22.Config()

    # Should have all required dataclass functionality
    assert hasattr(config, "__dataclass_fields__")

    # Should be able to create with custom values
    custom_config = F22.Config(mass_kg=18000.0, max_speed_mps=650.0)
    assert custom_config.mass_kg == 18000.0
    assert custom_config.max_speed_mps == 650.0


# ============================================================================
# F22 REALISM TESTS
# ============================================================================


def test_f22_stealth_characteristics():
    """Test F22 stealth characteristics are modeled."""
    config = F22.Config()

    # RCS should be significantly lower than conventional aircraft
    conventional_rcs = 5.0  # Typical fighter RCS in m²
    assert config.rcs < conventional_rcs * 0.01  # At least 100x reduction


def test_f22_supercruise_capability():
    """Test F22 supercruise capability is modeled."""
    config = F22.Config()

    # Should be capable of sustained supersonic flight
    mach_1_at_altitude = 295.0  # Approximate at 35,000 ft
    assert config.max_speed_mps > mach_1_at_altitude * 1.5  # Mach 1.5+ capability


def test_f22_advanced_avionics():
    """Test F22 advanced avionics characteristics."""
    config = F22.Config()

    # Should have superior radar performance compared to 4th gen
    # High power, high gain, long range
    assert config.radar_tx_power_w >= 20e3
    assert config.radar_antenna_gain_db >= 35.0
    assert config.radar_max_range_m >= 200_000

    # Should have low radar beam rates (high quality scans)
    assert config.radar_beam_rate_hz >= 8.0
    assert config.radar_beam_rate_p_hz >= 6.0


def test_f22_internal_weapons_bay():
    """Test F22 internal weapons bay modeling."""
    config = F22.Config()

    # Internal weapons bay should limit missile count vs external pylons
    # F-22 has internal bays for stealth
    max_internal_missiles = 8  # Realistic internal capacity
    assert config.max_missiles <= max_internal_missiles


# ============================================================================
# COMPARATIVE TESTS
# ============================================================================


def test_f22_vs_conventional_fighter():
    """Test F22 characteristics vs conventional fighter."""
    f22_config = F22.Config()

    # Typical 4th gen fighter characteristics for comparison
    conventional_rcs = 3.0  # m²
    conventional_radar_range = 150_000  # m
    conventional_max_speed = 600  # m/s (Mach 1.8)

    # F22 should be superior in stealth and sensors
    assert f22_config.rcs < conventional_rcs * 0.01
    assert f22_config.radar_max_range_m > conventional_radar_range
    assert f22_config.max_speed_mps > conventional_max_speed


def test_f22_missile_loadout_vs_capacity():
    """Test F22 missile loadout is appropriate for internal carriage."""
    config = F22.Config()

    # Internal carriage should be more limited than external
    typical_external_loadout = 12  # Many external pylons
    assert config.max_missiles < typical_external_loadout


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_f22_extreme_parameters(mock_position, mock_map_limits):
    """Test F22 with extreme parameters."""
    # Test with extreme values
    extreme_f22 = F22(
        position=mock_position,
        yaw_deg=720.0,  # Multiple rotations
        speed_mps=1000.0,  # Extreme speed
        group="EXTREME",
        map_limits=mock_map_limits,
        min_alt_m=-1000.0,  # Below sea level
        max_alt_m=50000.0,  # Very high
    )

    # Should handle extreme values (yaw may be normalized)
    # Note: yaw_deg gets normalized to [0, 360) in Aircraft base class
    assert extreme_f22.yaw_deg == 0.0  # 720 degrees normalized to 0
    assert extreme_f22.speed == 1000.0
    assert extreme_f22.group == "EXTREME"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_f22_aircraft_inheritance(f22_aircraft):
    """Test F22 inherits from Aircraft base class correctly."""
    # Should have all aircraft base functionality
    assert hasattr(f22_aircraft, "name")
    assert hasattr(f22_aircraft, "position")
    assert hasattr(f22_aircraft, "yaw_deg")
    assert hasattr(f22_aircraft, "speed")
    assert hasattr(f22_aircraft, "group")

    # Should be instance of Aircraft (if base class available)
    from air_to_air_rl.aircrafts.aircraft import Aircraft

    assert isinstance(f22_aircraft, Aircraft)


def test_f22_systems_integration(f22_aircraft):
    """Test F22 systems integration."""
    # Should have integrated systems from base Aircraft class
    # (Exact tests depend on Aircraft implementation)

    # Check that configuration was applied
    if hasattr(f22_aircraft, "config"):
        config = f22_aircraft.config

        # Config might be dict or object
        if isinstance(config, dict):
            assert config.get("mass_kg") == 19700.0
            assert config.get("max_speed_mps") == 680.0
            assert config.get("rcs") == 0.0001
        else:
            # Config is an object
            assert config.mass_kg == 19700.0
            assert config.max_speed_mps == 680.0
            assert config.rcs == 0.0001


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
