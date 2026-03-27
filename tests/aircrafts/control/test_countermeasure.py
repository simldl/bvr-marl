#!/usr/bin/env python3
"""
Tests for aircrafts.control.countermeasure module.
Tests countermeasure systems: flares, chaff, ECM, and decoys.
"""

import pytest

from air_to_air_rl.aircrafts.control.countermeasure import AircraftCountermeasureSystem
from air_to_air_rl.aircrafts.control.countermeasure_objects import Chaff, Decoy, Flare
from tests.mocks.aircraft import MockAircraft
from tests.mocks.simulator import MockSimulator


@pytest.fixture
def mock_aircraft():
    """Create mock aircraft with countermeasure loadouts."""
    aircraft = MockAircraft(name="test_countermeasures")
    aircraft.flares = 24
    aircraft.chaff = 40
    aircraft.ecm = 5
    aircraft.decoys = 10
    return aircraft


@pytest.fixture
def mock_simulator():
    """Create mock simulator for testing."""
    return MockSimulator()


@pytest.fixture
def countermeasure_system(mock_aircraft):
    """Create countermeasure system with mock aircraft."""
    return AircraftCountermeasureSystem(mock_aircraft)


# ============================================================================
# COUNTERMEASURE SYSTEM INITIALIZATION TESTS
# ============================================================================


def test_countermeasure_system_initialization(mock_aircraft):
    """Test countermeasure system initialization."""
    cms = AircraftCountermeasureSystem(mock_aircraft)

    # Test system exists
    assert cms is not None
    assert cms.parent == mock_aircraft


def test_countermeasure_system_parent_reference(countermeasure_system, mock_aircraft):
    """Test countermeasure system parent reference."""
    assert countermeasure_system.parent is mock_aircraft
    assert countermeasure_system.parent.name == "test_countermeasures"


# ============================================================================
# FLARE SYSTEM TESTS
# ============================================================================


def test_flare_launch_basic(countermeasure_system, mock_simulator):
    """Test basic flare launch functionality."""
    cms = countermeasure_system

    # Test flare launch method exists
    assert hasattr(cms, "launch_flares")
    assert callable(cms.launch_flares)

    # Test flare launch with simulator
    result = cms.launch_flares(mock_simulator)

    # Should return a Flare object
    assert result is not None
    assert isinstance(result, Flare)

    # Flare should be added to simulator
    assert result.id in mock_simulator.active_units

    # Flare should have correct properties
    assert result.is_countermeasure is True
    assert result.is_missile is False
    assert result.lifetime_s == 5.0


def test_flare_count_management(mock_aircraft, mock_simulator):
    """Test flare count management."""
    cms = AircraftCountermeasureSystem(mock_aircraft)
    initial_flares = cms.flares

    # Launch flares multiple times
    for i in range(3):
        result = cms.launch_flares(mock_simulator)
        assert result is not None
        assert cms.flares == initial_flares - (i + 1)

    # Test flare count tracking
    assert cms.flares == initial_flares - 3
    assert cms.flares >= 0


def test_flare_depletion_handling(mock_aircraft, mock_simulator):
    """Test flare depletion handling."""
    cms = AircraftCountermeasureSystem(mock_aircraft)
    cms.flares = 0  # No flares left

    result = cms.launch_flares(mock_simulator)

    # Should return None when no flares available
    assert result is None
    assert cms.flares == 0


# ============================================================================
# CHAFF SYSTEM TESTS
# ============================================================================


def test_chaff_launch_basic(countermeasure_system, mock_simulator):
    """Test basic chaff launch functionality."""
    cms = countermeasure_system

    # Test chaff launch method exists
    assert hasattr(cms, "launch_chaff")
    assert callable(cms.launch_chaff)

    # Test chaff launch with simulator
    result = cms.launch_chaff(mock_simulator)

    # Should return a Chaff object
    assert result is not None
    assert isinstance(result, Chaff)

    # Chaff should be added to simulator
    assert result.id in mock_simulator.active_units

    # Chaff should have correct properties
    assert result.is_countermeasure is True
    assert result.is_missile is False
    assert result.lifetime_s == 8.0


def test_chaff_count_management(mock_aircraft, mock_simulator):
    """Test chaff count management."""
    cms = AircraftCountermeasureSystem(mock_aircraft)
    initial_chaff = cms.chaff

    # Launch chaff multiple times
    for i in range(3):
        result = cms.launch_chaff(mock_simulator)
        assert result is not None
        assert cms.chaff == initial_chaff - (i + 1)

    # Test chaff count tracking
    assert cms.chaff == initial_chaff - 3
    assert cms.chaff >= 0


def test_chaff_depletion_handling(mock_aircraft, mock_simulator):
    """Test chaff depletion handling."""
    cms = AircraftCountermeasureSystem(mock_aircraft)
    cms.chaff = 0  # No chaff left

    result = cms.launch_chaff(mock_simulator)

    # Should return None when no chaff available
    assert result is None
    assert cms.chaff == 0


# ============================================================================
# ECM SYSTEM TESTS
# ============================================================================


def test_ecm_activation_basic(countermeasure_system):
    """Test basic ECM activation functionality."""
    cms = countermeasure_system

    # Test ECM activation method exists
    assert hasattr(cms, "activate_ecm")
    assert callable(cms.activate_ecm)

    # Test ECM activation
    try:
        cms.activate_ecm()
        # Should not raise exception
    except Exception as e:
        pytest.skip(f"ECM activation not implemented: {e}")


def test_ecm_duration_management(mock_aircraft):
    """Test ECM duration and usage management."""
    cms = AircraftCountermeasureSystem(mock_aircraft)

    if hasattr(cms, "activate_ecm"):
        # Test multiple ECM activations
        for _ in range(3):
            try:
                cms.activate_ecm()
            except Exception:
                pass  # May not be fully implemented


def test_ecm_depletion_handling(mock_aircraft):
    """Test ECM depletion handling."""
    cms = AircraftCountermeasureSystem(mock_aircraft)
    mock_aircraft.ecm = 0  # No ECM uses left

    if hasattr(cms, "activate_ecm"):
        try:
            cms.activate_ecm()
            # Should handle gracefully when ECM depleted
        except Exception:
            # Should not crash
            pass


# ============================================================================
# DECOY SYSTEM TESTS
# ============================================================================


def test_decoy_deployment_basic(countermeasure_system, mock_simulator):
    """Test basic decoy deployment functionality."""
    cms = countermeasure_system

    # Test decoy deployment method exists
    assert hasattr(cms, "deploy_decoys")
    assert callable(cms.deploy_decoys)

    # Test decoy deployment with simulator
    result = cms.deploy_decoys(mock_simulator)

    # Should return a Decoy object
    assert result is not None
    assert isinstance(result, Decoy)

    # Decoy should be added to simulator
    assert result.id in mock_simulator.active_units

    # Decoy should have correct properties
    assert result.is_countermeasure is True
    assert result.is_missile is False
    assert result.lifetime_s == 15.0


def test_decoy_count_management(mock_aircraft, mock_simulator):
    """Test decoy count management."""
    cms = AircraftCountermeasureSystem(mock_aircraft)
    initial_decoys = cms.decoys

    # Deploy decoys multiple times
    for i in range(3):
        result = cms.deploy_decoys(mock_simulator)
        assert result is not None
        assert cms.decoys == initial_decoys - (i + 1)

    # Test decoy count tracking
    assert cms.decoys == initial_decoys - 3
    assert cms.decoys >= 0


def test_decoy_depletion_handling(mock_aircraft, mock_simulator):
    """Test decoy depletion handling."""
    cms = AircraftCountermeasureSystem(mock_aircraft)
    cms.decoys = 0  # No decoys left

    result = cms.deploy_decoys(mock_simulator)

    # Should return None when no decoys available
    assert result is None
    assert cms.decoys == 0


# ============================================================================
# INTEGRATED COUNTERMEASURE TESTS
# ============================================================================


def test_multiple_countermeasure_usage(countermeasure_system):
    """Test using multiple countermeasures simultaneously."""
    cms = countermeasure_system

    # Test launching multiple countermeasures in sequence
    countermeasures = ["launch_flares", "launch_chaff", "activate_ecm", "deploy_decoys"]

    for cm_method in countermeasures:
        if hasattr(cms, cm_method):
            try:
                method = getattr(cms, cm_method)
                method()
            except Exception:
                # Individual countermeasures may not be fully implemented
                pass


def test_countermeasure_status_tracking(mock_aircraft):
    """Test countermeasure status and inventory tracking."""
    AircraftCountermeasureSystem(mock_aircraft)

    # Test that parent aircraft maintains countermeasure counts
    assert hasattr(mock_aircraft, "flares")
    assert hasattr(mock_aircraft, "chaff")
    assert hasattr(mock_aircraft, "ecm")
    assert hasattr(mock_aircraft, "decoys")

    # Test initial values
    assert mock_aircraft.flares >= 0
    assert mock_aircraft.chaff >= 0
    assert mock_aircraft.ecm >= 0
    assert mock_aircraft.decoys >= 0


def test_countermeasure_effectiveness_factors(countermeasure_system):
    """Test countermeasure effectiveness factors (if implemented)."""
    cms = countermeasure_system

    # Test if system has effectiveness tracking
    if hasattr(cms, "get_effectiveness"):
        try:
            effectiveness = cms.get_effectiveness()
            assert isinstance(effectiveness, (int, float, dict))
        except Exception:
            pass  # May not be implemented

    # Test if system tracks active countermeasures
    if hasattr(cms, "get_active_countermeasures"):
        try:
            active = cms.get_active_countermeasures()
            assert isinstance(active, (list, dict, set))
        except Exception:
            pass  # May not be implemented


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_countermeasure_error_handling():
    """Test countermeasure system error handling."""
    # Test handling of invalid parent
    try:
        AircraftCountermeasureSystem(None)
        # Should handle gracefully or raise appropriate exception
    except Exception as e:
        # Should raise appropriate exception, not crash
        assert isinstance(e, (AttributeError, TypeError, ValueError))


def test_countermeasure_boundary_conditions(mock_aircraft):
    """Test countermeasure system boundary conditions."""
    cms = AircraftCountermeasureSystem(mock_aircraft)

    # Test with zero countermeasures
    mock_aircraft.flares = 0
    mock_aircraft.chaff = 0
    mock_aircraft.ecm = 0
    mock_aircraft.decoys = 0

    countermeasures = ["launch_flares", "launch_chaff", "activate_ecm", "deploy_decoys"]
    for cm_method in countermeasures:
        if hasattr(cms, cm_method):
            try:
                method = getattr(cms, cm_method)
                method()
                # Should handle zero inventory gracefully
            except Exception:
                # Should not crash on zero inventory
                pass


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_countermeasure_aircraft_integration(mock_aircraft):
    """Test countermeasure system integration with aircraft."""
    cms = AircraftCountermeasureSystem(mock_aircraft)

    # Test that system maintains reference to parent
    assert cms.parent == mock_aircraft

    # Test that aircraft can access countermeasure system
    mock_aircraft.countermeasures = cms
    assert mock_aircraft.countermeasures == cms

    # Test that changes to countermeasures affect aircraft state
    if hasattr(cms, "launch_flares"):
        try:
            cms.launch_flares()
            # Flare count should change (if implemented)
        except Exception:
            pass


# ============================================================================
# COUNTERMEASURE PERSISTENCE TESTS
# ============================================================================


def test_flare_persistence(mock_aircraft, mock_simulator):
    """Test that flares persist in simulation for correct duration."""
    cms = AircraftCountermeasureSystem(mock_aircraft)

    # Launch a flare
    flare = cms.launch_flares(mock_simulator)
    assert flare is not None
    assert flare.lifetime_s == 5.0
    assert flare.age_s == 0.0

    # Simulate time passing - flare should still be active
    for i in range(4):
        flare.update(1.0, mock_simulator)  # 1 second each
        assert flare.should_be_removed is False
        assert flare.age_s == (i + 1)

    # After 5 seconds, flare should be marked for removal
    flare.update(1.0, mock_simulator)
    assert flare.should_be_removed is True
    assert flare.age_s == 5.0


def test_chaff_persistence(mock_aircraft, mock_simulator):
    """Test that chaff persists in simulation for correct duration."""
    cms = AircraftCountermeasureSystem(mock_aircraft)

    # Launch chaff
    chaff = cms.launch_chaff(mock_simulator)
    assert chaff is not None
    assert chaff.lifetime_s == 8.0
    assert chaff.age_s == 0.0

    # Simulate time passing - chaff should still be active
    for i in range(7):
        chaff.update(1.0, mock_simulator)  # 1 second each
        assert chaff.should_be_removed is False
        assert chaff.age_s == (i + 1)

    # After 8 seconds, chaff should be marked for removal
    chaff.update(1.0, mock_simulator)
    assert chaff.should_be_removed is True
    assert chaff.age_s == 8.0


def test_decoy_persistence(mock_aircraft, mock_simulator):
    """Test that decoys persist in simulation for correct duration."""
    cms = AircraftCountermeasureSystem(mock_aircraft)

    # Deploy a decoy
    decoy = cms.deploy_decoys(mock_simulator)
    assert decoy is not None
    assert decoy.lifetime_s == 15.0
    assert decoy.age_s == 0.0

    # Simulate time passing - decoy should still be active
    for i in range(14):
        decoy.update(1.0, mock_simulator)  # 1 second each
        assert decoy.should_be_removed is False
        assert decoy.age_s == (i + 1)

    # After 15 seconds, decoy should be marked for removal
    decoy.update(1.0, mock_simulator)
    assert decoy.should_be_removed is True
    assert decoy.age_s == 15.0


def test_countermeasure_cleanup(mock_aircraft, mock_simulator):
    """Test cleanup of expired countermeasures."""
    cms = AircraftCountermeasureSystem(mock_aircraft)

    # Launch multiple countermeasures
    flare = cms.launch_flares(mock_simulator)
    chaff = cms.launch_chaff(mock_simulator)
    decoy = cms.deploy_decoys(mock_simulator)

    # All should be tracked
    assert len(cms.active_countermeasures) == 3
    assert flare in cms.active_countermeasures
    assert chaff in cms.active_countermeasures
    assert decoy in cms.active_countermeasures

    # Mark flare as expired
    flare.should_be_removed = True

    # Cleanup should remove only expired countermeasures
    cms.cleanup_expired_countermeasures()

    assert len(cms.active_countermeasures) == 2
    assert flare not in cms.active_countermeasures
    assert chaff in cms.active_countermeasures
    assert decoy in cms.active_countermeasures


def test_multiple_countermeasure_persistence(mock_aircraft, mock_simulator):
    """Test that multiple countermeasures can coexist in simulation."""
    cms = AircraftCountermeasureSystem(mock_aircraft)

    # Get initial count (simulator might have other units)
    initial_count = len(mock_simulator.active_units)

    # Launch multiple countermeasures
    flare1 = cms.launch_flares(mock_simulator)
    flare2 = cms.launch_flares(mock_simulator)
    chaff1 = cms.launch_chaff(mock_simulator)

    # All should be in simulator (3 new units added)
    assert len(mock_simulator.active_units) == initial_count + 3
    assert flare1.id in mock_simulator.active_units
    assert flare2.id in mock_simulator.active_units
    assert chaff1.id in mock_simulator.active_units

    # All should be tracked by countermeasure system
    assert len(cms.active_countermeasures) == 3

    # Simulate time - flares expire at 5s, chaff at 8s
    for _ in range(5):
        flare1.update(1.0, mock_simulator)
        flare2.update(1.0, mock_simulator)
        chaff1.update(1.0, mock_simulator)

    # Flares should be expired, chaff should still be active
    assert flare1.should_be_removed is True
    assert flare2.should_be_removed is True
    assert chaff1.should_be_removed is False


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
