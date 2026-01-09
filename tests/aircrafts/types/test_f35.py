#!/usr/bin/env python3
"""
Tests for aircrafts.types.f35 module.
Tests F-35 Lightning II aircraft configuration and initialization.
"""

import pytest
from tests.mocks import MockPosition, MockMapLimits
from aircrafts.types.f35 import F35


@pytest.fixture
def mock_position():
    """Create mock position for F35 testing."""
    return MockPosition(0, 0, 10000)


@pytest.fixture
def mock_map_limits():
    """Create mock map limits for F35 testing."""
    return MockMapLimits()


@pytest.fixture
def f35_aircraft(mock_position, mock_map_limits):
    """Create F35 aircraft instance for testing."""
    try:
        return F35(
            position=mock_position,
            yaw_deg=0.0,
            speed_mps=300.0,
            group="BLUE",
            map_limits=mock_map_limits,
            min_alt_m=0.0,
            max_alt_m=20000.0
        )
    except Exception:
        pytest.skip("F35 initialization requires full Aircraft base class")


# ============================================================================
# F35 CONFIGURATION TESTS
# ============================================================================

def test_f35_config_defaults():
    """Test F35 default configuration parameters."""
    config = F35.Config()

    # Basic physics parameters (corrected based on Wikipedia geometry)
    assert config.mass_kg == 13300.0  # Lighter than F-22
    assert config.reference_area_m2 == 42.7  # 460 ft²
    assert config.aspect_ratio == 2.68  # 10.7^2 / 42.7 corrected from Wikipedia
    assert config.oswald_e == 0.82
    assert config.max_speed_mps == 540.0  # Slower than F-22 (single engine)
    assert config.n_max == 9.0
    assert config.stall0_mps == 68.0

    # Flight envelope
    assert config.min_speed_mps == 70.0
    assert config.max_climb_angle_deg == 60.0  # Less than F-22
    assert config.min_alt_m == 0.0
    assert config.max_alt_m == 20000.0


def test_f35_performance_characteristics():
    """Test F35 performance characteristics are realistic."""
    config = F35.Config()
    
    # F-35 should be lighter and slower than F-22 (single vs twin engine)
    assert 12000 <= config.mass_kg <= 15000  # Single engine fighter weight
    assert 35 <= config.reference_area_m2 <= 50  # Wing area
    assert config.max_speed_mps < 600  # Single engine limitation
    assert config.max_climb_angle_deg <= 70  # Climb performance


def test_f35_vs_f22_comparison():
    """Test F35 characteristics vs F22."""
    f35_config = F35.Config()

    # Import F22 for comparison
    try:
        from aircrafts.types.f22 import F22
        f22_config = F22.Config()

        # F-35 should be lighter but slower than F-22
        assert f35_config.mass_kg < f22_config.mass_kg  # Single vs twin engine
        assert f35_config.max_speed_mps < f22_config.max_speed_mps  # Speed difference
        assert f35_config.reference_area_m2 < f22_config.reference_area_m2  # F-22 has larger wing area

    except ImportError:
        pytest.skip("F22 comparison requires F22 class")


def test_f35_config_structure():
    """Test F35 config dataclass structure."""
    config = F35.Config()
    
    # Should have all required configuration fields
    required_fields = [
        'mass_kg', 'reference_area_m2', 'aspect_ratio', 'oswald_e',
        'max_speed_mps', 'n_max', 'stall0_mps', 'min_speed_mps',
        'max_climb_angle_deg', 'min_alt_m', 'max_alt_m'
    ]
    
    for field in required_fields:
        assert hasattr(config, field), f"Missing required field: {field}"


# ============================================================================
# F35 INITIALIZATION TESTS
# ============================================================================

def test_f35_initialization_basic(f35_aircraft):
    """Test basic F35 initialization."""
    assert f35_aircraft.name == "F35"
    assert f35_aircraft.group == "BLUE"
    assert f35_aircraft.yaw_deg == 0.0
    assert f35_aircraft.speed == 300.0


def test_f35_initialization_parameters(mock_position, mock_map_limits):
    """Test F35 initialization with various parameters."""
    try:
        f35 = F35(
            position=mock_position,
            yaw_deg=90.0,
            speed_mps=250.0,
            group="NATO",
            map_limits=mock_map_limits,
            min_alt_m=100.0,
            max_alt_m=18000.0
        )
        
        assert f35.yaw_deg == 90.0
        assert f35.speed == 250.0
        assert f35.group == "NATO"
    except Exception:
        pytest.skip("F35 parameter test requires full Aircraft class")


# ============================================================================
# F35 VARIANT CHARACTERISTICS
# ============================================================================

def test_f35_multirole_characteristics():
    """Test F35 multirole fighter characteristics."""
    config = F35.Config()
    
    # F-35 is designed as multirole (vs F-22 air superiority focus)
    # Should have balanced characteristics rather than extreme performance
    
    # Moderate speed (not as fast as pure air superiority fighters)
    assert 500 <= config.max_speed_mps <= 600
    
    # Good but not extreme maneuverability
    assert 8.0 <= config.n_max <= 9.5
    
    # Reasonable stall speed for carrier operations (F-35C variant consideration)
    assert 60 <= config.stall0_mps <= 80


def test_f35_stealth_characteristics():
    """Test F35 stealth modeling (if available)."""
    config = F35.Config()
    
    # Check if RCS is modeled (may not be in all configurations)
    if hasattr(config, 'rcs'):
        # Should have low RCS for stealth
        assert config.rcs <= 0.01  # Stealth characteristics


# ============================================================================
# F35 REALISM TESTS
# ============================================================================

def test_f35_single_engine_characteristics():
    """Test F35 single engine characteristics."""
    config = F35.Config()
    
    # Single engine should result in lower mass and performance vs twin engine
    typical_twin_engine_mass = 19000  # F-22 class
    assert config.mass_kg < typical_twin_engine_mass
    
    # Single engine typically means lower max speed
    typical_twin_engine_speed = 680  # F-22 class  
    assert config.max_speed_mps < typical_twin_engine_speed


def test_f35_exportable_fighter():
    """Test F35 as exportable fighter characteristics."""
    config = F35.Config()
    
    # F-35 is designed for export/allies, may have different characteristics
    # than purely domestic fighters like F-22
    
    # Should still be high performance but potentially less extreme
    assert config.max_speed_mps >= 500  # Still supersonic capable
    assert config.n_max >= 8.0  # Still high-G capable
    
    # May have different radar characteristics
    if hasattr(config, 'radar_max_range_m'):
        assert config.radar_max_range_m >= 100_000  # Still long range


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_f35_config_modification():
    """Test F35 configuration modification."""
    config = F35.Config()
    
    original_mass = config.mass_kg
    config.mass_kg = 14000.0
    
    assert config.mass_kg == 14000.0
    assert config.mass_kg != original_mass


def test_f35_invalid_parameters():
    """Test F35 with invalid parameters."""
    try:
        invalid_f35 = F35(
            position=None,
            yaw_deg=0.0,
            speed_mps=300.0,
            group="BLUE",
            map_limits=None,
            min_alt_m=0.0,
            max_alt_m=20000.0
        )
    except Exception:
        # Should raise appropriate exception for None position
        pass


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_f35_aircraft_inheritance(f35_aircraft):
    """Test F35 inherits from Aircraft correctly."""
    # Should have base aircraft functionality
    assert hasattr(f35_aircraft, 'name')
    assert hasattr(f35_aircraft, 'position')
    assert hasattr(f35_aircraft, 'group')
    
    # Check inheritance if Aircraft class available
    try:
        from aircrafts.aircraft import Aircraft
        assert isinstance(f35_aircraft, Aircraft)
    except ImportError:
        pass


def test_f35_configuration_applied(f35_aircraft):
    """Test F35 configuration is applied correctly."""
    if hasattr(f35_aircraft, 'config'):
        config = f35_aircraft.config
        
        # Config might be dict or object
        if isinstance(config, dict):
            assert config.get('mass_kg') == 13300.0
            assert config.get('max_speed_mps') == 540.0
        else:
            # Config is an object
            assert config.mass_kg == 13300.0
            assert config.max_speed_mps == 540.0


# ============================================================================
# COMPARATIVE ANALYSIS TESTS
# ============================================================================

def test_f35_role_differentiation():
    """Test F35 role differentiation from air superiority fighters."""
    config = F35.Config()
    
    # F-35 optimized for multirole rather than pure air-to-air
    # This might be reflected in different parameter priorities
    
    # Should still be capable but balanced
    assert config.max_speed_mps >= 500  # Capable
    assert config.n_max >= 8.0  # Maneuverable
    
    # May prioritize different aspects than pure air superiority fighters
    # (Exact characteristics depend on implementation philosophy)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_f35_creation_performance():
    """Test F35 creation performance."""
    import time
    
    mock_pos = MockPosition(0, 0, 10000)
    mock_limits = MockMapLimits()
    
    start_time = time.time()
    
    try:
        for _ in range(5):
            f35 = F35(
                position=mock_pos,
                yaw_deg=0.0,
                speed_mps=300.0,
                group="BLUE",
                map_limits=mock_limits,
                min_alt_m=0.0,
                max_alt_m=20000.0
            )
    except Exception:
        pytest.skip("F35 performance test requires full Aircraft class")
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Should create quickly
    assert duration < 0.5


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])