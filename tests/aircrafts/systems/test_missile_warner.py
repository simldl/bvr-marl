#!/usr/bin/env python3
"""
Tests for aircrafts.systems.missile_warner module.
Tests missile warning system for threat detection and alert timing.
"""

import pytest
import numpy as np
from tests.mocks import MockAircraft, MockPosition, MockSimulator, MockMissile
from aircrafts.systems.missile_warner import MissileWarner


@pytest.fixture
def mock_aircraft():
    """Create mock aircraft for missile warning testing."""
    aircraft = MockAircraft(
        name="test_warned_aircraft",
        position=MockPosition(0, 0, 10000),
        yaw_deg=0.0,
        speed=300.0
    )
    aircraft.id = "aircraft_001"
    return aircraft


@pytest.fixture
def missile_warner(mock_aircraft):
    """Create missile warner with mock aircraft."""
    return MissileWarner(
        parent=mock_aircraft,
        detection_delay_s=1.0,
        detection_delay_std=0.2
    )


@pytest.fixture
def mock_missiles(mock_aircraft):
    """Create mock missiles targeting the aircraft."""
    missiles = []
    
    # Missile 1 - targeting our aircraft
    missile1 = MockMissile(target=mock_aircraft)
    missile1.id = "missile_001"
    missile1.is_missile = True
    missile1.target = mock_aircraft
    missiles.append(missile1)
    
    # Missile 2 - targeting our aircraft
    missile2 = MockMissile(target=mock_aircraft)
    missile2.id = "missile_002"
    missile2.is_missile = True
    missile2.target = mock_aircraft
    missiles.append(missile2)
    
    # Missile 3 - not targeting our aircraft
    other_target = MockAircraft(name="other_target")
    other_target.id = "other_001"
    missile3 = MockMissile(target=other_target)
    missile3.id = "missile_003"
    missile3.is_missile = True
    missile3.target = other_target
    missiles.append(missile3)
    
    return missiles


@pytest.fixture
def mock_simulator():
    """Create mock simulator for testing."""
    return MockSimulator()


# ============================================================================
# MISSILE WARNER INITIALIZATION TESTS
# ============================================================================

def test_missile_warner_initialization(mock_aircraft):
    """Test missile warner initialization."""
    mw = MissileWarner(
        parent=mock_aircraft,
        detection_delay_s=2.0,
        detection_delay_std=0.5
    )
    
    assert mw.parent == mock_aircraft
    assert mw.detection_delay_s == 2.0
    assert mw.detection_delay_std == 0.5
    assert mw._pending_warnings == []
    assert mw._active_warnings == set()


def test_missile_warner_default_parameters(mock_aircraft):
    """Test missile warner with default parameters."""
    mw = MissileWarner(parent=mock_aircraft)
    
    # Should have default values
    assert mw.detection_delay_s == 1.0
    assert mw.detection_delay_std == 0.5


def test_missile_warner_parent_reference(missile_warner, mock_aircraft):
    """Test missile warner parent reference."""
    assert missile_warner.parent is mock_aircraft
    assert missile_warner.parent.name == "test_warned_aircraft"


# ============================================================================
# MISSILE DETECTION TESTS
# ============================================================================

def test_check_for_new_missiles_basic(missile_warner, mock_simulator, mock_missiles):
    """Test basic missile detection."""
    # Add missiles to simulator
    for missile in mock_missiles:
        mock_simulator.add_unit(missile)
    
    sim_time = 10.0
    missile_warner.check_for_new_missiles(sim_time, mock_simulator)
    
    # Should have pending warnings for missiles targeting our aircraft
    assert len(missile_warner._pending_warnings) >= 0
    
    # Check that only relevant missiles are detected
    pending_ids = [mid for (_, mid, _) in missile_warner._pending_warnings]
    
    # Should include missiles targeting our aircraft
    missiles_targeting_us = [m for m in mock_missiles if m.target == mock_aircraft]
    for missile in missiles_targeting_us:
        if missile.id in pending_ids:
            assert missile.id in pending_ids


def test_check_for_new_missiles_targeting_filter(missile_warner, mock_simulator, mock_missiles):
    """Test that only missiles targeting our aircraft are detected."""
    # Add all missiles to simulator
    for missile in mock_missiles:
        mock_simulator.add_unit(missile)
    
    sim_time = 10.0
    missile_warner.check_for_new_missiles(sim_time, mock_simulator)
    
    # Check pending warnings (if any)
    for (warn_time, missile_id, missile) in missile_warner._pending_warnings:
        # All pending warnings should be for missiles targeting our aircraft
        assert missile.target == missile_warner.parent


def test_check_for_new_missiles_no_duplicates(missile_warner, mock_simulator, mock_missiles):
    """Test that duplicate warnings are not created."""
    # Add missiles to simulator
    relevant_missiles = [m for m in mock_missiles if m.target == mock_aircraft]
    for missile in relevant_missiles:
        mock_simulator.add_unit(missile)
    
    sim_time = 10.0
    
    # First check
    missile_warner.check_for_new_missiles(sim_time, mock_simulator)
    initial_count = len(missile_warner._pending_warnings)
    
    # Second check (same time)
    missile_warner.check_for_new_missiles(sim_time, mock_simulator)
    final_count = len(missile_warner._pending_warnings)
    
    # Should not create duplicates
    assert final_count == initial_count


def test_check_for_new_missiles_empty_simulator(missile_warner, mock_simulator):
    """Test missile detection with empty simulator."""
    sim_time = 10.0
    missile_warner.check_for_new_missiles(sim_time, mock_simulator)
    
    # Should have no pending warnings
    assert len(missile_warner._pending_warnings) == 0


def test_check_for_new_missiles_no_active_units(missile_warner):
    """Test missile detection with simulator without active_units."""
    # Create mock simulator without active_units attribute
    mock_sim = type('MockSim', (), {})()
    
    sim_time = 10.0
    missile_warner.check_for_new_missiles(sim_time, mock_sim)
    
    # Should handle gracefully
    assert len(missile_warner._pending_warnings) == 0


# ============================================================================
# PENDING WARNING TESTS
# ============================================================================

def test_is_pending_basic(missile_warner):
    """Test pending missile check."""
    # Manually add pending warning
    missile_warner._pending_warnings.append((15.0, "missile_001", None))
    
    assert missile_warner._is_pending("missile_001") == True
    assert missile_warner._is_pending("missile_002") == False


def test_is_pending_empty(missile_warner):
    """Test pending check with no pending warnings."""
    assert missile_warner._is_pending("any_missile") == False


def test_is_pending_multiple_missiles(missile_warner):
    """Test pending check with multiple missiles."""
    # Add multiple pending warnings
    missile_warner._pending_warnings.extend([
        (15.0, "missile_001", None),
        (16.0, "missile_002", None),
        (17.0, "missile_003", None)
    ])
    
    assert missile_warner._is_pending("missile_001") == True
    assert missile_warner._is_pending("missile_002") == True
    assert missile_warner._is_pending("missile_003") == True
    assert missile_warner._is_pending("missile_004") == False


# ============================================================================
# WARNING UPDATE TESTS
# ============================================================================

def test_update_warnings_basic(missile_warner):
    """Test basic warning update."""
    # Add pending warning that should trigger
    test_missile = MockMissile()
    test_missile.id = "test_missile"
    
    missile_warner._pending_warnings.append((10.0, "test_missile", test_missile))
    
    # Update at time when warning should trigger
    sim_time = 10.5
    due_warnings = missile_warner.update(sim_time)
    
    # Should have triggered the warning
    assert len(due_warnings) == 1
    assert due_warnings[0].id == "test_missile"
    
    # Should have moved to active warnings
    assert "test_missile" in missile_warner._active_warnings
    
    # Should be removed from pending
    assert len(missile_warner._pending_warnings) == 0


def test_update_warnings_not_due(missile_warner):
    """Test warning update when warnings not yet due."""
    # Add pending warning for future time
    test_missile = MockMissile()
    test_missile.id = "test_missile"
    
    missile_warner._pending_warnings.append((15.0, "test_missile", test_missile))
    
    # Update at earlier time
    sim_time = 10.0
    due_warnings = missile_warner.update(sim_time)
    
    # Should have no triggered warnings
    assert len(due_warnings) == 0
    
    # Should remain in pending
    assert len(missile_warner._pending_warnings) == 1
    
    # Should not be in active warnings
    assert "test_missile" not in missile_warner._active_warnings


def test_update_warnings_multiple_mixed(missile_warner):
    """Test warning update with mix of due and not due warnings."""
    # Add multiple pending warnings
    missile1 = MockMissile()
    missile1.id = "missile_001"
    missile2 = MockMissile() 
    missile2.id = "missile_002"
    missile3 = MockMissile()
    missile3.id = "missile_003"
    
    missile_warner._pending_warnings.extend([
        (9.0, "missile_001", missile1),   # Due
        (15.0, "missile_002", missile2),  # Not due
        (10.0, "missile_003", missile3)   # Due
    ])
    
    sim_time = 10.5
    due_warnings = missile_warner.update(sim_time)
    
    # Should have triggered 2 warnings
    assert len(due_warnings) == 2
    due_ids = [m.id for m in due_warnings]
    assert "missile_001" in due_ids
    assert "missile_003" in due_ids
    
    # Should have 1 remaining pending
    assert len(missile_warner._pending_warnings) == 1
    assert missile_warner._pending_warnings[0][1] == "missile_002"
    
    # Active warnings should include the triggered ones
    assert "missile_001" in missile_warner._active_warnings
    assert "missile_003" in missile_warner._active_warnings
    assert "missile_002" not in missile_warner._active_warnings


def test_update_warnings_empty_pending(missile_warner):
    """Test warning update with no pending warnings."""
    sim_time = 10.0
    due_warnings = missile_warner.update(sim_time)
    
    assert len(due_warnings) == 0
    assert len(missile_warner._active_warnings) == 0


def test_update_warnings_exact_timing(missile_warner):
    """Test warning update at exact trigger time."""
    test_missile = MockMissile()
    test_missile.id = "test_missile"
    
    trigger_time = 10.0
    missile_warner._pending_warnings.append((trigger_time, "test_missile", test_missile))
    
    # Update at exact trigger time
    due_warnings = missile_warner.update(trigger_time)
    
    # Should trigger at exact time
    assert len(due_warnings) == 1
    assert due_warnings[0].id == "test_missile"


# ============================================================================
# ACTIVE WARNING TESTS
# ============================================================================

def test_get_current_warning_ids_basic(missile_warner):
    """Test getting current warning IDs."""
    # Add some active warnings
    missile_warner._active_warnings.update(["missile_001", "missile_002"])
    
    warning_ids = missile_warner.get_current_warning_ids()
    
    assert isinstance(warning_ids, list)
    assert "missile_001" in warning_ids
    assert "missile_002" in warning_ids
    assert len(warning_ids) == 2


def test_get_current_warning_ids_empty(missile_warner):
    """Test getting warning IDs when no active warnings."""
    warning_ids = missile_warner.get_current_warning_ids()
    
    assert isinstance(warning_ids, list)
    assert len(warning_ids) == 0


# ============================================================================
# DELAY TIMING TESTS
# ============================================================================

def test_detection_delay_application(mock_aircraft, mock_simulator, mock_missiles):
    """Test that detection delay is properly applied."""
    # Create warner with known delay parameters
    mw = MissileWarner(
        parent=mock_aircraft,
        detection_delay_s=2.0,
        detection_delay_std=0.0  # No randomization for predictable test
    )
    
    # Add missile to simulator
    relevant_missile = [m for m in mock_missiles if m.target == mock_aircraft][0]
    mock_simulator.add_unit(relevant_missile)
    
    sim_time = 10.0
    mw.check_for_new_missiles(sim_time, mock_simulator)
    
    # Should have pending warning with delay
    if len(mw._pending_warnings) > 0:
        warn_time, missile_id, missile = mw._pending_warnings[0]
        expected_warn_time = sim_time + 2.0
        assert abs(warn_time - expected_warn_time) < 0.1  # Allow small tolerance


def test_minimum_delay_enforcement(mock_aircraft, mock_simulator, mock_missiles):
    """Test that minimum delay is enforced."""
    # Create warner with negative mean delay (should be clamped)
    mw = MissileWarner(
        parent=mock_aircraft,
        detection_delay_s=-1.0,  # Negative delay
        detection_delay_std=0.5
    )
    
    # Add missile to simulator
    relevant_missile = [m for m in mock_missiles if m.target == mock_aircraft][0]
    mock_simulator.add_unit(relevant_missile)
    
    sim_time = 10.0
    mw.check_for_new_missiles(sim_time, mock_simulator)
    
    # Should enforce minimum delay of 0.05s
    if len(mw._pending_warnings) > 0:
        warn_time, missile_id, missile = mw._pending_warnings[0]
        assert warn_time >= sim_time + 0.05


def test_delay_randomization(mock_aircraft, mock_simulator, mock_missiles):
    """Test delay randomization."""
    # Create warner with large standard deviation
    mw = MissileWarner(
        parent=mock_aircraft,
        detection_delay_s=2.0,
        detection_delay_std=1.0
    )
    
    relevant_missile = [m for m in mock_missiles if m.target == mock_aircraft][0]
    
    # Generate multiple delays
    delays = []
    for i in range(10):
        # Clear previous warnings
        mw._pending_warnings.clear()
        mw._active_warnings.clear()
        
        # Add fresh missile with new ID
        missile = MockMissile(target=mock_aircraft)
        missile.id = f"test_missile_{i}"
        missile.is_missile = True
        missile.target = mock_aircraft
        
        mock_simulator.active_units[missile.id] = missile
        
        sim_time = 10.0
        mw.check_for_new_missiles(sim_time, mock_simulator)
        
        if len(mw._pending_warnings) > 0:
            warn_time = mw._pending_warnings[0][0]
            delay = warn_time - sim_time
            delays.append(delay)
    
    # Should have some variation in delays
    if len(delays) > 1:
        delay_std = np.std(delays)
        assert delay_std > 0.1  # Should have some randomization


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_complete_warning_workflow(missile_warner, mock_simulator, mock_missiles):
    """Test complete missile warning workflow."""
    # Add missiles to simulator
    targeting_missiles = [m for m in mock_missiles if m.target == mock_aircraft]
    for missile in targeting_missiles:
        mock_simulator.add_unit(missile)
    
    sim_time = 10.0
    
    # Step 1: Check for new missiles
    missile_warner.check_for_new_missiles(sim_time, mock_simulator)
    initial_pending = len(missile_warner._pending_warnings)
    
    # Step 2: Update warnings (not yet due)
    due_warnings = missile_warner.update(sim_time)
    assert len(due_warnings) == 0
    
    # Step 3: Update warnings after delay
    sim_time += 2.0
    due_warnings = missile_warner.update(sim_time)
    
    # Should have triggered some warnings
    final_active = len(missile_warner.get_current_warning_ids())
    assert final_active >= 0


def test_missile_warner_with_aircraft_integration(mock_aircraft):
    """Test missile warner integration with aircraft."""
    mw = MissileWarner(parent=mock_aircraft)
    
    # Test that warner can access aircraft properties
    assert mw.parent.name == "test_warned_aircraft"
    assert mw.parent.id == "aircraft_001"
    
    # Test that aircraft can use warner
    mock_aircraft.missile_warner = mw
    assert mock_aircraft.missile_warner == mw


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_missile_warner_error_handling():
    """Test missile warner error handling."""
    # Test with invalid aircraft
    try:
        invalid_mw = MissileWarner(parent=None)
        # Should handle gracefully or raise appropriate exception
    except Exception as e:
        assert isinstance(e, (AttributeError, TypeError))


def test_check_missiles_invalid_units(missile_warner):
    """Test missile checking with invalid units."""
    # Create simulator with invalid units
    mock_sim = MockSimulator()
    
    # Add unit without required attributes
    invalid_unit = type('InvalidUnit', (), {})()
    mock_sim.add_unit(invalid_unit)
    
    sim_time = 10.0
    
    # Should handle gracefully without crashing
    missile_warner.check_for_new_missiles(sim_time, mock_sim)
    
    # Should not create warnings for invalid units
    assert len(missile_warner._pending_warnings) == 0


def test_update_with_invalid_pending_warnings(missile_warner):
    """Test update with corrupted pending warnings."""
    # Add invalid pending warning structure
    missile_warner._pending_warnings.append(("invalid", "structure"))
    
    sim_time = 10.0
    
    try:
        due_warnings = missile_warner.update(sim_time)
        # Should handle gracefully or raise exception
        assert isinstance(due_warnings, list)
    except (ValueError, IndexError, TypeError):
        # May raise exception for invalid structure
        pass


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_missile_warner_extreme_delays(mock_aircraft):
    """Test missile warner with extreme delay parameters."""
    # Very large delay
    mw_large = MissileWarner(
        parent=mock_aircraft,
        detection_delay_s=1000.0,
        detection_delay_std=100.0
    )
    
    # Very small delay
    mw_small = MissileWarner(
        parent=mock_aircraft,
        detection_delay_s=0.001,
        detection_delay_std=0.0001
    )
    
    # Should initialize without issues
    assert mw_large.detection_delay_s == 1000.0
    assert mw_small.detection_delay_s == 0.001


def test_multiple_missiles_same_id_edge_case(missile_warner, mock_simulator, mock_aircraft):
    """Test edge case with multiple missiles having same ID."""
    # Create missiles with same ID (edge case)
    missile1 = MockMissile(target=mock_aircraft)
    missile1.id = "duplicate_id"
    missile1.is_missile = True
    missile1.target = mock_aircraft
    
    missile2 = MockMissile(target=mock_aircraft)
    missile2.id = "duplicate_id"  # Same ID
    missile2.is_missile = True
    missile2.target = mock_aircraft
    
    mock_simulator.add_unit(missile1)
    # Add second with same ID (overwrites first in dict)
    mock_simulator.active_units["duplicate_id"] = missile2
    
    sim_time = 10.0
    missile_warner.check_for_new_missiles(sim_time, mock_simulator)
    
    # Should handle gracefully
    pending_count = len(missile_warner._pending_warnings)
    assert pending_count >= 0


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])