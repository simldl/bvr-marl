import json
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

from bvr_marl_core.domain.information import (
    FrameReference,
    SensorReport,
    SensorType,
    TrackLifecycle,
    TrackSnapshot,
    WeaponTrack,
)


def _snapshot(time_s=2.0):
    return TrackSnapshot(
        track_id=17,
        state_time_s=time_s,
        state=(100.0, 200.0, 300.0, 10.0, -5.0, 2.0),
        covariance=np.eye(6),
        confidence=0.8,
        lifecycle=TrackLifecycle.CONFIRMED,
        source_ids=(3, 4),
        report_lineage=((3, 1), (4, 1)),
    )


def test_track_snapshot_serializes_without_live_objects():
    snapshot = _snapshot()

    payload = json.loads(json.dumps(asdict(snapshot)))

    assert payload["track_id"] == 17
    assert payload["lifecycle"] == "confirmed"
    assert payload["report_lineage"] == [[3, 1], [4, 1]]


def test_sensor_report_is_immutable_and_copies_mutable_inputs():
    measurement = np.array([1.0, 2.0, 3.0])
    covariance = np.eye(3)
    metadata = {"snr_db": 12.0}
    report = SensorReport(
        report_id=1,
        source_id=2,
        acquisition_time_s=3.0,
        measurement=measurement,
        covariance=covariance,
        frame=FrameReference(48.0, 11.0, 5000.0),
        sensor_type=SensorType.RADAR,
        classification_probabilities=(0.7, 0.1, 0.05, 0.05, 0.1),
        metadata=metadata,
    )

    measurement[0] = 99.0
    covariance[0, 0] = 99.0
    metadata["snr_db"] = -1.0

    assert report.measurement == (1.0, 2.0, 3.0)
    assert report.covariance[0][0] == 1.0
    assert report.metadata["snr_db"] == 12.0
    assert report["snr_db"] == 12.0
    assert report["report_id"] == 1
    assert report.classification_schema_version == 1
    assert report.lineage == ((2, 1),)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        report.report_id = 9
    with pytest.raises(TypeError):
        report.metadata["snr_db"] = 0.0


def test_report_rejects_invalid_covariance():
    with pytest.raises(ValueError, match="symmetric"):
        SensorReport(
            report_id=1,
            source_id=2,
            acquisition_time_s=3.0,
            measurement=(1.0, 2.0),
            covariance=((1.0, 2.0), (0.0, 1.0)),
            frame=FrameReference(0.0, 0.0, 0.0),
            sensor_type=SensorType.RADAR,
        )


def test_track_snapshot_projects_roundoff_scale_negative_eigenvalue():
    covariance = np.eye(6) * 1.0e8
    covariance[-1, -1] = -1.0e-6

    snapshot = TrackSnapshot(
        track_id=1,
        state_time_s=0.0,
        state=(0.0,) * 6,
        covariance=covariance,
        confidence=1.0,
        lifecycle=TrackLifecycle.CONFIRMED,
    )

    assert np.min(np.linalg.eigvalsh(snapshot.covariance)) >= 0.0


def test_track_snapshot_rejects_materially_indefinite_covariance():
    covariance = np.eye(6)
    covariance[-1, -1] = -0.1
    with pytest.raises(ValueError, match="positive semidefinite"):
        TrackSnapshot(
            track_id=1,
            state_time_s=0.0,
            state=(0.0,) * 6,
            covariance=covariance,
            confidence=1.0,
            lifecycle=TrackLifecycle.CONFIRMED,
        )


def test_report_rejects_wrong_classification_schema_width():
    with pytest.raises(ValueError, match="requires 5 probabilities"):
        SensorReport(
            report_id=1,
            source_id=2,
            acquisition_time_s=3.0,
            measurement=(1.0, 2.0, 3.0),
            covariance=np.eye(3),
            frame=FrameReference(0.0, 0.0, 0.0),
            sensor_type=SensorType.RADAR,
            classification_probabilities=(0.5, 0.5),
        )


def test_weapon_track_coasts_without_mutating_last_snapshot():
    original = _snapshot()
    weapon_track = WeaponTrack(original, launch_time_s=2.0)

    coasted = weapon_track.estimate_at(5.0)

    assert coasted.state[:3] == pytest.approx((130.0, 185.0, 306.0))
    assert coasted.lifecycle is TrackLifecycle.COASTING
    assert coasted.age_s == pytest.approx(3.0)
    assert coasted.report_lineage == ((3, 1), (4, 1))
    assert original.state[:3] == (100.0, 200.0, 300.0)


def test_track_snapshot_canonicalizes_source_qualified_lineage():
    snapshot = TrackSnapshot(
        track_id=1,
        state_time_s=0.0,
        state=(0.0,) * 6,
        covariance=np.eye(6),
        confidence=1.0,
        lifecycle=TrackLifecycle.CONFIRMED,
        report_lineage=(("sensor-b", 1), ("sensor-a", 1), ("sensor-a", 1)),
    )

    assert snapshot.report_lineage == (("sensor-a", 1), ("sensor-b", 1))


def test_track_coast_covariance_contains_constant_acceleration_uncertainty():
    coasted = _snapshot(time_s=0.0).propagated(15.0, maneuver_accel_std_mps2=10.0)
    covariance = np.asarray(coasted.covariance)

    expected_position_variance = 0.25 * 10.0**2 * 15.0**4
    expected_velocity_variance = 10.0**2 * 15.0**2
    expected_cross_covariance = 0.5 * 10.0**2 * 15.0**3
    assert covariance[0, 0] >= expected_position_variance
    assert covariance[3, 3] >= expected_velocity_variance
    assert covariance[0, 3] >= expected_cross_covariance


def test_weapon_track_rejects_identity_switch_and_stale_update():
    weapon_track = WeaponTrack(_snapshot(), launch_time_s=2.0)
    switched = TrackSnapshot(
        track_id=99,
        state_time_s=3.0,
        state=(0.0,) * 6,
        covariance=np.eye(6),
        confidence=1.0,
        lifecycle=TrackLifecycle.CONFIRMED,
    )
    with pytest.raises(ValueError, match="identity"):
        weapon_track.with_update(switched)
    with pytest.raises(ValueError, match="backward"):
        weapon_track.with_update(_snapshot(time_s=1.0))
