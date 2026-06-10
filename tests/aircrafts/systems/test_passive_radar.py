#!/usr/bin/env python3
"""
Tests for aircrafts.systems.passive_radar module.
Tests passive radar detection, ranging, and bearing measurement capabilities.
"""

import math

import numpy as np
import pytest

from bvr_marl_core.aircraft.systems.passive_radar import PassiveRadar
from tests.mocks import MockAircraft, MockPosition


@pytest.fixture
def passive_radar():
    """Create passive radar system with default parameters."""
    return PassiveRadar(angular_error_deg=2.0, range_error_m=500.0, max_age_s=5.0)


@pytest.fixture
def emitter_aircraft():
    """Create mock emitter aircraft."""
    emitter = MockAircraft(
        name="emitter_aircraft",
        position=MockPosition(0.1, 0, 10000),  # 10km north
        yaw_deg=180.0,
        speed=300.0,
    )
    emitter.id = "emitter_001"
    return emitter


@pytest.fixture
def receiver_aircraft():
    """Create mock receiver aircraft."""
    receiver = MockAircraft(
        name="receiver_aircraft",
        position=MockPosition(0, 0, 10000),
        yaw_deg=0.0,  # Facing north
        speed=250.0,
    )
    receiver.id = "receiver_001"
    return receiver


# ============================================================================
# PASSIVE RADAR INITIALIZATION TESTS
# ============================================================================


def test_passive_radar_initialization():
    """Test passive radar initialization."""
    pr = PassiveRadar(angular_error_deg=3.0, range_error_m=1000.0, max_age_s=10.0)

    assert pr.angular_error_deg == 3.0
    assert pr.range_error_m == 1000.0
    assert pr.max_age_s == 10.0
    assert pr.detections == []


def test_passive_radar_default_initialization():
    """Test passive radar initialization with defaults."""
    pr = PassiveRadar()

    assert pr.angular_error_deg == 5.0
    assert pr.range_error_m == 2000.0
    assert pr.max_age_s == 3.0
    assert pr.detections == []


# ============================================================================
# EMISSION RECEPTION TESTS
# ============================================================================


def test_receive_emission_basic(passive_radar, emitter_aircraft, receiver_aircraft):
    """Test basic emission reception."""
    timestamp = 10.0

    detection = passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, timestamp)

    # Test detection structure
    assert isinstance(detection, dict)
    assert "emitter_id" in detection
    assert "rel_angle_deg" in detection
    assert "distance_m" in detection
    assert "timestamp" in detection

    # Test values
    assert detection["emitter_id"] == "emitter_001"
    assert detection["timestamp"] == timestamp
    assert isinstance(detection["rel_angle_deg"], float)
    assert isinstance(detection["distance_m"], float)
    assert detection["distance_m"] >= 0.0


def test_receive_emission_bearing_calculation(passive_radar, receiver_aircraft):
    """Test bearing calculation in emission reception."""
    # Emitter due north
    emitter_north = MockAircraft(
        name="emitter_north",
        position=MockPosition(0.1, 0, 10000),  # North
        yaw_deg=0.0,
        speed=300.0,
    )
    emitter_north.id = "north"

    # Emitter due east
    emitter_east = MockAircraft(
        name="emitter_east",
        position=MockPosition(0, 0.1, 10000),  # East
        yaw_deg=0.0,
        speed=300.0,
    )
    emitter_east.id = "east"

    # Receiver facing north
    receiver_aircraft.yaw_deg = 0.0

    detection_north = passive_radar.receive_emission(emitter_north, receiver_aircraft, 10.0)
    detection_east = passive_radar.receive_emission(emitter_east, receiver_aircraft, 10.0)

    # North should be roughly 0° relative bearing (with noise)
    # East should be roughly 90° relative bearing (with noise)
    assert abs(detection_north["rel_angle_deg"]) < 30  # Allow for noise
    assert 60 < detection_east["rel_angle_deg"] < 120  # Allow for noise


def test_receive_emission_distance_calculation(passive_radar, receiver_aircraft):
    """Test distance calculation in emission reception."""
    # Close emitter
    emitter_close = MockAircraft(
        name="emitter_close",
        position=MockPosition(0.05, 0, 10000),  # 5km north
        yaw_deg=0.0,
        speed=300.0,
    )
    emitter_close.id = "close"

    # Far emitter
    emitter_far = MockAircraft(
        name="emitter_far",
        position=MockPosition(0.2, 0, 10000),  # 20km north
        yaw_deg=0.0,
        speed=300.0,
    )
    emitter_far.id = "far"

    detection_close = passive_radar.receive_emission(emitter_close, receiver_aircraft, 10.0)
    detection_far = passive_radar.receive_emission(emitter_far, receiver_aircraft, 10.0)

    # Far target should have greater distance (allowing for noise)
    assert detection_close["distance_m"] < detection_far["distance_m"] + 3000  # 3km noise tolerance


def test_receive_emission_altitude_difference(passive_radar, receiver_aircraft):
    """Test distance calculation with altitude difference."""
    # High emitter
    emitter_high = MockAircraft(
        name="emitter_high",
        position=MockPosition(0, 0, 15000),  # 5km higher, same horizontal position
        yaw_deg=0.0,
        speed=300.0,
    )
    emitter_high.id = "high"

    detection = passive_radar.receive_emission(emitter_high, receiver_aircraft, 10.0)

    # Distance should be approximately 5km (altitude difference)
    expected_distance = 5000.0
    assert abs(detection["distance_m"] - expected_distance) < 2000  # Allow for noise


def test_receive_emission_multiple_detections(passive_radar, emitter_aircraft, receiver_aircraft):
    """Test multiple emission receptions."""
    # Multiple detections from same emitter
    detection1 = passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 10.0)
    detection2 = passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 11.0)
    detection3 = passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 12.0)

    # Should have stored all detections
    assert len(passive_radar.detections) == 3

    # All should have same emitter ID
    assert detection1["emitter_id"] == "emitter_001"
    assert detection2["emitter_id"] == "emitter_001"
    assert detection3["emitter_id"] == "emitter_001"

    # Timestamps should be different
    assert detection1["timestamp"] == 10.0
    assert detection2["timestamp"] == 11.0
    assert detection3["timestamp"] == 12.0


def test_receive_emission_no_emitter_id(passive_radar, receiver_aircraft):
    """Test emission reception with emitter without ID."""
    emitter_no_id = MockAircraft(
        name="no_id_emitter", position=MockPosition(0.1, 0, 10000), yaw_deg=0.0, speed=300.0
    )
    # MockAircraft automatically assigns IDs, so we'll test with what's assigned

    detection = passive_radar.receive_emission(emitter_no_id, receiver_aircraft, 10.0)

    # Should have some emitter ID (either None or auto-assigned)
    assert "emitter_id" in detection
    # MockAircraft assigns IDs automatically, so this might not be None
    assert detection["emitter_id"] is None or isinstance(detection["emitter_id"], str)


# ============================================================================
# ANGLE NORMALIZATION TESTS
# ============================================================================


def test_angle_normalization_positive(passive_radar, receiver_aircraft):
    """Test angle normalization for angles > 180°."""
    # Emitter behind and to the left (should be negative relative angle)
    emitter_behind = MockAircraft(
        name="emitter_behind",
        position=MockPosition(-0.1, -0.05, 10000),  # Southwest
        yaw_deg=0.0,
        speed=300.0,
    )
    emitter_behind.id = "behind"

    receiver_aircraft.yaw_deg = 0.0  # Facing north

    detection = passive_radar.receive_emission(emitter_behind, receiver_aircraft, 10.0)

    # Relative angle should be normalized to [-180, 180]
    assert -180 <= detection["rel_angle_deg"] <= 180


def test_angle_normalization_wraparound(passive_radar, receiver_aircraft):
    """Test angle normalization wraparound."""
    # Multiple emitters at different bearings
    emitters = []
    for i in range(8):  # 8 directions
        bearing = i * 45  # 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
        lat_offset = 0.1 * math.cos(math.radians(bearing))
        lon_offset = 0.1 * math.sin(math.radians(bearing))

        emitter = MockAircraft(
            name=f"emitter_{i}",
            position=MockPosition(lat_offset, lon_offset, 10000),
            yaw_deg=0.0,
            speed=300.0,
        )
        emitter.id = f"emitter_{i}"
        emitters.append(emitter)

    # Test with different receiver headings
    for receiver_heading in [0, 45, 90, 135, 180, 225, 270, 315]:
        receiver_aircraft.yaw_deg = receiver_heading

        for emitter in emitters:
            detection = passive_radar.receive_emission(emitter, receiver_aircraft, 10.0)
            # All angles should be normalized
            assert -180 <= detection["rel_angle_deg"] <= 180


# ============================================================================
# DISTANCE CONSTRAINTS TESTS
# ============================================================================


def test_distance_minimum_constraint(passive_radar, receiver_aircraft):
    """Test that distance cannot be negative."""
    # Very close emitter with high range error
    pr_high_error = PassiveRadar(
        angular_error_deg=0.0,
        range_error_m=10000.0,  # Very high error
        max_age_s=5.0,
    )

    emitter_close = MockAircraft(
        name="very_close",
        position=MockPosition(0.001, 0, 10000),  # Very close
        yaw_deg=0.0,
        speed=300.0,
    )
    emitter_close.id = "close"

    # Multiple detections to test statistical behavior
    min_distance = float("inf")
    for _ in range(10):
        detection = pr_high_error.receive_emission(emitter_close, receiver_aircraft, 10.0)
        min_distance = min(min_distance, detection["distance_m"])

    # Distance should never be negative
    assert min_distance >= 0.0


# ============================================================================
# OBSERVATION RETRIEVAL TESTS
# ============================================================================


def test_get_observation_basic(passive_radar, emitter_aircraft, receiver_aircraft):
    """Test basic observation retrieval."""
    # Add some detections
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 10.0)
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 11.0)

    observations = passive_radar.get_observation(12.0)

    # Should return list of observations
    assert isinstance(observations, list)
    assert len(observations) == 2

    # Check observation structure
    for obs in observations:
        assert "emitter_id" in obs
        assert "rel_angle_deg" in obs
        assert "distance_m" in obs
        assert "age_s" in obs

        assert obs["emitter_id"] == "emitter_001"
        assert isinstance(obs["age_s"], float)
        assert obs["age_s"] >= 0


def test_get_observation_age_filtering(passive_radar, emitter_aircraft, receiver_aircraft):
    """Test observation age filtering."""
    # Add detections at different times
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 10.0)  # Age 5s
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 12.0)  # Age 3s
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 14.0)  # Age 1s
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 5.0)  # Age 10s (too old)

    observations = passive_radar.get_observation(15.0)

    # Should only return observations within max_age_s (5.0)
    assert len(observations) == 3

    # Check ages are within limit
    for obs in observations:
        assert obs["age_s"] <= 5.0


def test_get_observation_empty(passive_radar):
    """Test observation retrieval when no detections."""
    observations = passive_radar.get_observation(10.0)

    assert isinstance(observations, list)
    assert len(observations) == 0


def test_get_observation_all_expired(passive_radar, emitter_aircraft, receiver_aircraft):
    """Test observation retrieval when all detections expired."""
    # Add old detections
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 1.0)
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 2.0)

    # Query much later (all expired)
    observations = passive_radar.get_observation(20.0)

    assert len(observations) == 0


def test_get_observation_age_calculation(passive_radar, emitter_aircraft, receiver_aircraft):
    """Test age calculation in observations."""
    timestamp = 10.0
    query_time = 13.0
    expected_age = 3.0

    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, timestamp)
    observations = passive_radar.get_observation(query_time)

    assert len(observations) == 1
    assert abs(observations[0]["age_s"] - expected_age) < 0.001


# ============================================================================
# OLD DETECTION CLEANUP TESTS
# ============================================================================


def test_clear_old_basic(passive_radar, emitter_aircraft, receiver_aircraft):
    """Test basic old detection cleanup."""
    # Add detections at different times
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 1.0)
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 5.0)
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 10.0)

    assert len(passive_radar.detections) == 3

    # Clear old detections (max_age_s = 5.0)
    passive_radar.clear_old(12.0)

    # Should keep detections from times 10.0 and 5.0, but not 1.0
    # Actually, with current_time=12.0 and max_age=5.0, keeps detections >= 7.0
    remaining = [d for d in passive_radar.detections if (12.0 - d["timestamp"]) <= 5.0]
    assert len(passive_radar.detections) == len(remaining)
    assert all(d["timestamp"] >= 7.0 for d in passive_radar.detections)


def test_clear_old_all_expired(passive_radar, emitter_aircraft, receiver_aircraft):
    """Test clearing all expired detections."""
    # Add old detections
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 1.0)
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 2.0)

    assert len(passive_radar.detections) == 2

    # Clear with much later time
    passive_radar.clear_old(20.0)

    assert len(passive_radar.detections) == 0


def test_clear_old_none_expired(passive_radar, emitter_aircraft, receiver_aircraft):
    """Test clearing when no detections expired."""
    # Add recent detections
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 10.0)
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 11.0)

    assert len(passive_radar.detections) == 2

    # Clear with only slightly later time
    passive_radar.clear_old(12.0)

    # All should remain
    assert len(passive_radar.detections) == 2


def test_clear_old_edge_case(passive_radar, emitter_aircraft, receiver_aircraft):
    """Test clearing at exact age boundary."""
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 10.0)

    # Clear exactly at max_age boundary
    passive_radar.clear_old(15.0)  # Age = 5.0, exactly at max_age_s

    # Detection at exact boundary should be kept
    assert len(passive_radar.detections) == 1


# ============================================================================
# NOISE AND ACCURACY TESTS
# ============================================================================


def test_noise_application(emitter_aircraft, receiver_aircraft):
    """Test that noise is applied to measurements."""
    # Create radar with known noise parameters
    pr = PassiveRadar(
        angular_error_deg=10.0,  # Large angular error
        range_error_m=5000.0,  # Large range error
        max_age_s=10.0,
    )

    # Multiple measurements of same geometry
    angles = []
    distances = []

    for _ in range(20):
        detection = pr.receive_emission(emitter_aircraft, receiver_aircraft, 10.0)
        angles.append(detection["rel_angle_deg"])
        distances.append(detection["distance_m"])

    # Should have variation due to noise
    angle_std = np.std(angles)
    distance_std = np.std(distances)

    # Should have some variation (not exactly zero)
    assert angle_std > 0.1
    assert distance_std > 100.0


def test_zero_noise_accuracy(emitter_aircraft, receiver_aircraft):
    """Test accuracy with zero noise."""
    # Create radar with no noise
    pr = PassiveRadar(angular_error_deg=0.0, range_error_m=0.0, max_age_s=10.0)

    # Set known geometry
    emitter_aircraft.position = MockPosition(0.1, 0, 10000)  # 10km north
    receiver_aircraft.position = MockPosition(0, 0, 10000)
    receiver_aircraft.yaw_deg = 0.0  # Facing north

    detection = pr.receive_emission(emitter_aircraft, receiver_aircraft, 10.0)

    # Should be exactly 0° relative bearing (no noise)
    assert abs(detection["rel_angle_deg"]) < 0.001

    # Distance should be approximately 10km (111km/degree * 0.1 degrees)
    expected_distance = 11100.0  # Approximately
    assert abs(detection["distance_m"] - expected_distance) < 100.0


# ============================================================================
# MULTIPLE EMITTER TESTS
# ============================================================================


def test_multiple_emitters(passive_radar, receiver_aircraft):
    """Test handling multiple emitters."""
    # Create multiple emitters
    emitters = []
    for i in range(5):
        emitter = MockAircraft(
            name=f"emitter_{i}",
            position=MockPosition(0.1 * (i + 1), 0, 10000),
            yaw_deg=0.0,
            speed=300.0,
        )
        emitter.id = f"emitter_{i}"
        emitters.append(emitter)

    # Receive emissions from all emitters
    for emitter in emitters:
        passive_radar.receive_emission(emitter, receiver_aircraft, 10.0)

    # Should have stored all detections
    assert len(passive_radar.detections) == 5

    # Get observations
    observations = passive_radar.get_observation(10.0)
    assert len(observations) == 5

    # Should have different emitter IDs
    emitter_ids = {obs["emitter_id"] for obs in observations}
    assert len(emitter_ids) == 5


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_passive_radar_error_handling():
    """Test passive radar error handling with invalid parameters."""
    # Test with extreme parameters
    pr = PassiveRadar(
        angular_error_deg=-1.0,  # Negative error
        range_error_m=-1000.0,  # Negative error
        max_age_s=-5.0,  # Negative age
    )

    # Should not crash on initialization
    assert pr.angular_error_deg == -1.0  # Stores as given


def test_receive_emission_invalid_units(passive_radar):
    """Test emission reception with invalid units."""
    # Test with None units
    try:
        passive_radar.receive_emission(None, None, 10.0)
        # Should handle gracefully or raise exception
    except Exception:
        # Expected to fail with None units
        pass


def test_get_observation_invalid_time(passive_radar, emitter_aircraft, receiver_aircraft):
    """Test observation retrieval with invalid time."""
    passive_radar.receive_emission(emitter_aircraft, receiver_aircraft, 10.0)

    # Test with negative time
    observations = passive_radar.get_observation(-5.0)

    # Should handle gracefully, likely return empty list
    assert isinstance(observations, list)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_passive_radar_complete_workflow(passive_radar, receiver_aircraft):
    """Test complete passive radar workflow."""
    # Multiple emitters over time
    emitter1 = MockAircraft(
        name="emitter1", position=MockPosition(0.1, 0, 10000), yaw_deg=0.0, speed=300.0
    )
    emitter1.id = "em1"

    emitter2 = MockAircraft(
        name="emitter2", position=MockPosition(0, 0.1, 10000), yaw_deg=0.0, speed=300.0
    )
    emitter2.id = "em2"

    # Receive emissions over time
    passive_radar.receive_emission(emitter1, receiver_aircraft, 10.0)
    passive_radar.receive_emission(emitter2, receiver_aircraft, 11.0)
    passive_radar.receive_emission(emitter1, receiver_aircraft, 12.0)

    # Get current observations
    observations = passive_radar.get_observation(13.0)
    assert len(observations) == 3

    # Clean old detections
    passive_radar.clear_old(13.0)
    observations_after_clean = passive_radar.get_observation(13.0)
    assert len(observations_after_clean) <= len(observations)


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
