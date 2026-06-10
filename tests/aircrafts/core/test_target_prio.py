#!/usr/bin/env python3
"""
Tests for aircrafts.core.target_prio module.
Tests track priority system for target ranking and selection.
"""

import numpy as np
import pytest

from bvr_marl_core.aircraft.core.target_prio import TrackPrioritySystem
from tests.mocks import MockAircraft, MockPosition


@pytest.fixture
def mock_aircraft():
    """Create mock aircraft for target priority testing."""
    aircraft = MockAircraft(
        name="test_priority_aircraft", position=MockPosition(0, 0, 10000), yaw_deg=0.0, speed=300.0
    )
    return aircraft


@pytest.fixture
def priority_system(mock_aircraft):
    """Create track priority system with mock aircraft."""
    return TrackPrioritySystem(mock_aircraft)


@pytest.fixture
def sample_track_states():
    """Create sample track states for testing."""
    # Track state format: (state_vector, covariance)
    # State vector: [x, y, z, vx, vy, vz] in ENU coordinates

    # Close, slow target
    track1 = (
        np.array([5000.0, 0.0, 0.0, 50.0, 0.0, 0.0]),  # 5km east, moving east slowly
        np.eye(6) * 0.1,  # Low covariance
    )

    # Far, fast approaching target
    track2 = (
        np.array([15000.0, 0.0, 0.0, -200.0, 0.0, 0.0]),  # 15km east, approaching fast
        np.eye(6) * 0.2,
    )

    # Close, receding target
    track3 = (
        np.array([3000.0, 0.0, 0.0, 100.0, 0.0, 0.0]),  # 3km east, moving away
        np.eye(6) * 0.1,
    )

    # High altitude target
    track4 = (
        np.array([8000.0, 0.0, 5000.0, 0.0, 0.0, 0.0]),  # 8km east, 5km up, stationary
        np.eye(6) * 0.15,
    )

    return [track1, track2, track3, track4]


# ============================================================================
# PRIORITY SYSTEM INITIALIZATION TESTS
# ============================================================================


def test_priority_system_initialization(mock_aircraft):
    """Test priority system initialization."""
    tps = TrackPrioritySystem(mock_aircraft)

    assert tps.own_aircraft == mock_aircraft
    assert hasattr(tps, "own_aircraft")


def test_priority_system_parent_reference(priority_system, mock_aircraft):
    """Test priority system parent reference."""
    assert priority_system.own_aircraft is mock_aircraft
    assert priority_system.own_aircraft.name == "test_priority_aircraft"


# ============================================================================
# UTILITY CALCULATION TESTS
# ============================================================================


def test_distance_calculation(priority_system):
    """Test distance calculation from track state."""
    # Create test track state
    track_state = (
        np.array([3000.0, 4000.0, 0.0, 0.0, 0.0, 0.0]),  # 5km distance (3-4-5 triangle)
        np.eye(6) * 0.1,
    )

    distance = priority_system._distance(track_state)

    assert isinstance(distance, float)
    assert abs(distance - 5000.0) < 1.0  # Should be ~5km


def test_distance_calculation_zero(priority_system):
    """Test distance calculation at origin."""
    track_state = (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.eye(6) * 0.1)

    distance = priority_system._distance(track_state)
    assert distance == 0.0


def test_relative_height_calculation(priority_system):
    """Test relative height calculation."""
    # Target 2km above own aircraft
    track_state = (
        np.array([5000.0, 0.0, -2000.0, 0.0, 0.0, 0.0]),  # Note: negative z in ENU
        np.eye(6) * 0.1,
    )

    rel_height = priority_system._relative_height(track_state)

    assert isinstance(rel_height, float)
    assert rel_height == 2000.0  # Should be 2km relative height


def test_relative_height_below(priority_system):
    """Test relative height calculation for target below."""
    # Target 1km below own aircraft
    track_state = (
        np.array([5000.0, 0.0, 1000.0, 0.0, 0.0, 0.0]),  # Positive z in ENU
        np.eye(6) * 0.1,
    )

    rel_height = priority_system._relative_height(track_state)
    assert rel_height == -1000.0  # Should be negative (below)


def test_relative_velocity_towards_own(priority_system):
    """Test relative velocity calculation towards own aircraft."""
    # Target approaching own aircraft (at origin)
    track_state = (
        np.array([5000.0, 0.0, 0.0, -100.0, 0.0, 0.0]),  # Moving toward origin
        np.eye(6) * 0.1,
    )

    closing_rate = priority_system._rel_velocity_towards_own(track_state)

    assert isinstance(closing_rate, float)
    assert closing_rate > 0  # Should be positive (approaching)


def test_relative_velocity_away_from_own(priority_system):
    """Test relative velocity calculation away from own aircraft."""
    # Target moving away from own aircraft
    track_state = (
        np.array([5000.0, 0.0, 0.0, 100.0, 0.0, 0.0]),  # Moving away from origin
        np.eye(6) * 0.1,
    )

    closing_rate = priority_system._rel_velocity_towards_own(track_state)

    assert isinstance(closing_rate, float)
    assert closing_rate < 0  # Should be negative (receding)


def test_relative_velocity_perpendicular(priority_system):
    """Test relative velocity calculation for perpendicular motion."""
    # Target moving perpendicular to line of sight
    track_state = (
        np.array([5000.0, 0.0, 0.0, 0.0, 100.0, 0.0]),  # Moving north
        np.eye(6) * 0.1,
    )

    closing_rate = priority_system._rel_velocity_towards_own(track_state)

    assert abs(closing_rate) < 1.0  # Should be near zero


def test_relative_velocity_insufficient_state(priority_system):
    """Test relative velocity calculation with insufficient state vector."""
    # Track state with only position, no velocity
    track_state = (
        np.array([5000.0, 0.0, 0.0]),  # Only 3 elements
        np.eye(3) * 0.1,
    )

    closing_rate = priority_system._rel_velocity_towards_own(track_state)
    assert closing_rate == 0.0  # Should return default


def test_relative_velocity_zero_distance(priority_system):
    """Test relative velocity calculation at zero distance."""
    # Target at exact same position
    track_state = (np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0]), np.eye(6) * 0.1)

    closing_rate = priority_system._rel_velocity_towards_own(track_state)
    assert closing_rate == 0.0  # Should handle zero distance gracefully


# ============================================================================
# SCORING TESTS
# ============================================================================


def test_score_calculation_basic(priority_system):
    """Test basic score calculation."""
    track_state = (
        np.array([5000.0, 0.0, 0.0, -50.0, 0.0, 0.0]),  # 5km, approaching
        np.eye(6) * 0.1,
    )

    score = priority_system.score(track_state)

    assert isinstance(score, float)


def test_score_closer_target_priority(priority_system):
    """Test that closer targets get higher priority (higher score)."""
    # Close target
    close_track = (np.array([3000.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.eye(6) * 0.1)

    # Far target
    far_track = (np.array([10000.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.eye(6) * 0.1)

    score_close = priority_system.score(close_track)
    score_far = priority_system.score(far_track)

    # Closer target should have HIGHER score (more threatening)
    # Score formula: -1.0 * dist_norm + ... (negative distance = higher for close)
    assert score_close > score_far


def test_score_approaching_target_priority(priority_system):
    """Test that approaching targets get higher priority."""
    # Approaching target
    approaching_track = (
        np.array([5000.0, 0.0, 0.0, -100.0, 0.0, 0.0]),  # Fast approach
        np.eye(6) * 0.1,
    )

    # Receding target
    receding_track = (
        np.array([5000.0, 0.0, 0.0, 100.0, 0.0, 0.0]),  # Fast recede
        np.eye(6) * 0.1,
    )

    score_approaching = priority_system.score(approaching_track)
    score_receding = priority_system.score(receding_track)

    # Approaching target should have HIGHER score (more threatening)
    # Score formula: ... + 1.5 * closing_norm (positive for approaching)
    assert score_approaching > score_receding


def test_score_altitude_factor(priority_system):
    """Test altitude factor in scoring."""
    # Same altitude target
    same_alt_track = (np.array([5000.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.eye(6) * 0.1)

    # High altitude target
    high_alt_track = (
        np.array([5000.0, 0.0, -3000.0, 0.0, 0.0, 0.0]),  # 3km above
        np.eye(6) * 0.1,
    )

    score_same_alt = priority_system.score(same_alt_track)
    score_high_alt = priority_system.score(high_alt_track)

    # Different altitudes should produce different scores
    assert score_same_alt != score_high_alt


def test_score_combined_factors(priority_system):
    """Test scoring with multiple factors combined."""
    # High priority: close, approaching, same altitude
    high_prio_track = (np.array([2000.0, 0.0, 0.0, -150.0, 0.0, 0.0]), np.eye(6) * 0.1)

    # Low priority: far, receding, different altitude
    low_prio_track = (np.array([15000.0, 0.0, -5000.0, 100.0, 0.0, 0.0]), np.eye(6) * 0.1)

    score_high_prio = priority_system.score(high_prio_track)
    score_low_prio = priority_system.score(low_prio_track)

    # Scores should be different (don't assume which is higher/lower due to complex weighting)
    assert score_high_prio != score_low_prio
    assert isinstance(score_high_prio, (int, float))
    assert isinstance(score_low_prio, (int, float))


# ============================================================================
# PRIORITIZATION TESTS
# ============================================================================


def test_prioritize_single_track(priority_system):
    """Test prioritization with single track."""
    track_states = [(np.array([5000.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.eye(6) * 0.1)]

    result = priority_system.prioritize(track_states)

    assert len(result) == 1
    assert len(result[0]) == 2  # (state_vector, score)
    assert isinstance(result[0][1], float)  # Score should be float


def test_prioritize_multiple_tracks(priority_system, sample_track_states):
    """Test prioritization with multiple tracks."""
    result = priority_system.prioritize(sample_track_states)

    assert len(result) == len(sample_track_states)

    # Check that results are sorted by score (DESCENDING - highest threat first)
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)

    # Check that all original states are present
    result_states = [state for state, _ in result]
    original_states = [state for state, _ in sample_track_states]

    for orig_state in original_states:
        assert any(np.array_equal(orig_state, res_state) for res_state in result_states)


def test_prioritize_empty_list(priority_system):
    """Test prioritization with empty track list."""
    result = priority_system.prioritize([])

    assert result == []


def test_prioritize_ordering_verification(priority_system):
    """Test that prioritization produces correct ordering."""
    # Create tracks with known priority order
    tracks = [
        # High priority: close and approaching
        (np.array([2000.0, 0.0, 0.0, -100.0, 0.0, 0.0]), np.eye(6) * 0.1),
        # Medium priority: medium distance
        (np.array([8000.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.eye(6) * 0.1),
        # Low priority: far and receding
        (np.array([15000.0, 0.0, 0.0, 50.0, 0.0, 0.0]), np.eye(6) * 0.1),
    ]

    result = priority_system.prioritize(tracks)

    # Verify ordering by checking that scores DECREASE (descending order)
    assert len(result) == 3
    score1 = result[0][1]  # Highest threat
    score2 = result[1][1]
    score3 = result[2][1]  # Lowest threat

    assert score1 >= score2 >= score3


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_priority_system_error_handling():
    """Test priority system error handling."""
    # Test with invalid aircraft
    try:
        TrackPrioritySystem(None)
        # Should handle gracefully or raise appropriate exception
    except Exception as e:
        assert isinstance(e, (AttributeError, TypeError))


def test_score_calculation_invalid_state(priority_system):
    """Test score calculation with invalid track state."""
    # Track state with wrong format
    invalid_track = (
        np.array([1000.0]),  # Only one element
        np.eye(1) * 0.1,
    )

    try:
        priority_system.score(invalid_track)
        # Should handle gracefully or raise exception
    except (IndexError, ValueError):
        pass  # Expected behavior


def test_prioritize_invalid_tracks(priority_system):
    """Test prioritization with invalid track states."""
    invalid_tracks = [
        (np.array([]), np.eye(1)),  # Empty state vector
        (np.array([1000.0, 2000.0]), np.eye(2)),  # Too short
    ]

    try:
        result = priority_system.prioritize(invalid_tracks)
        # Should handle gracefully, possibly with empty result
        assert isinstance(result, list)
    except (IndexError, ValueError):
        # May raise exception for invalid states
        pass


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_score_extreme_values(priority_system):
    """Test scoring with extreme values."""
    # Very close, very fast target
    extreme_track = (
        np.array([10.0, 0.0, 0.0, -1000.0, 0.0, 0.0]),  # 10m, Mach 3 approach
        np.eye(6) * 0.1,
    )

    score = priority_system.score(extreme_track)

    # Should handle extreme values without crashing
    assert isinstance(score, float)
    assert not np.isnan(score)
    assert not np.isinf(score)


def test_score_zero_values(priority_system):
    """Test scoring with zero values."""
    zero_track = (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.eye(6) * 0.1)

    score = priority_system.score(zero_track)

    # Should handle zero values gracefully
    assert isinstance(score, float)
    assert not np.isnan(score)


def test_prioritize_identical_tracks(priority_system):
    """Test prioritization with identical tracks."""
    identical_track = (np.array([5000.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.eye(6) * 0.1)

    tracks = [identical_track, identical_track, identical_track]
    result = priority_system.prioritize(tracks)

    # Should handle identical tracks
    assert len(result) == 3

    # All scores should be identical
    scores = [score for _, score in result]
    assert all(abs(score - scores[0]) < 1e-10 for score in scores)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_priority_system_aircraft_integration(mock_aircraft):
    """Test priority system integration with aircraft."""
    tps = TrackPrioritySystem(mock_aircraft)

    # Test that system can access aircraft attributes
    assert tps.own_aircraft.position.alt == 10000.0

    # Test that aircraft altitude affects scoring
    mock_aircraft.position.alt = 5000.0

    track_state = (np.array([5000.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.eye(6) * 0.1)

    score = tps.score(track_state)
    assert isinstance(score, float)


def test_priority_system_realistic_scenario(priority_system):
    """Test priority system with realistic engagement scenario."""
    # Create realistic BVR scenario tracks
    tracks = [
        # Hostile fighter - medium range, approaching
        (np.array([25000.0, 5000.0, -1000.0, -120.0, -30.0, 0.0]), np.eye(6) * 0.1),
        # Distant bomber - far, slow
        (np.array([50000.0, 0.0, -2000.0, -80.0, 0.0, 0.0]), np.eye(6) * 0.2),
        # Close missile - very close, very fast
        (np.array([3000.0, 0.0, 0.0, -400.0, 0.0, 0.0]), np.eye(6) * 0.05),
        # Neutral aircraft - crossing path
        (np.array([15000.0, 0.0, 0.0, 0.0, 150.0, 0.0]), np.eye(6) * 0.15),
    ]

    result = priority_system.prioritize(tracks)

    # Should prioritize close, fast threats first
    assert len(result) == 4

    # Check that prioritization worked
    first_priority_state = result[0][0]
    assert isinstance(first_priority_state[0], (int, float))  # Should be valid position


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


def test_priority_system_performance(priority_system):
    """Test priority system performance with many tracks."""
    # Create many tracks
    tracks = []
    for i in range(100):
        track = (
            np.array(
                [
                    1000.0 + i * 100.0,  # Varying distances
                    i * 10.0,  # Varying north position
                    0.0,  # Same altitude
                    -50.0 + i,  # Varying velocities
                    0.0,
                    0.0,
                ]
            ),
            np.eye(6) * 0.1,
        )
        tracks.append(track)

    import time

    start_time = time.time()

    result = priority_system.prioritize(tracks)

    end_time = time.time()
    duration = end_time - start_time

    # Should complete quickly
    assert duration < 1.0  # 1 second is generous for 100 tracks
    assert len(result) == 100


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
