from types import SimpleNamespace

import numpy as np

from bvr_marl_core.domain.information import TrackLifecycle
from bvr_marl_core.radar.tracking.tracker import TrackerManager
from bvr_marl_core.simulator.core.helpers import Position


def _cluster(north_m, report_id, source_id=3):
    return {
        "az": 0.0,
        "el": 0.0,
        "d": north_m,
        "dop": 0.0,
        "source_pos": Position(0.0, 0.0, 0.0),
        "report_ids": (report_id,),
        "source_ids": (source_id,),
        "report_lineage": ((source_id, report_id),),
        "n_obs": 1,
    }


def test_anonymous_reports_keep_track_identity_through_motion():
    manager = TrackerManager(assoc_dist=100.0)
    reference = Position(0.0, 0.0, 0.0)

    first = manager.update_tracks([_cluster(10_000.0, 1)], 1.0, reference)
    second = manager.update_tracks([_cluster(10_300.0, 2)], 1.0, reference)
    third = manager.update_tracks([_cluster(10_600.0, 3)], 1.0, reference)

    assert len(first) == len(second) == len(third) == 1
    assert first[0].track_id == second[0].track_id == third[0].track_id
    assert third[0].engageable is True


def test_imm_tracker_runs_production_anonymous_update_path():
    manager = TrackerManager(assoc_dist=100.0, motion_model="imm_cv_ct")
    reference = Position(0.0, 0.0, 0.0)

    snapshots = []
    for report_id, north_m in enumerate((10_000.0, 10_300.0, 10_600.0), start=1):
        snapshots = manager.update_tracks([_cluster(north_m, report_id)], 1.0, reference)

    assert len(snapshots) == 1
    assert snapshots[0].engageable is True
    assert np.all(np.isfinite(snapshots[0].state))
    assert np.linalg.eigvalsh(snapshots[0].covariance).min() >= -1e-8


def test_global_assignment_does_not_depend_on_report_order():
    reference = Position(0.0, 0.0, 0.0)
    manager = TrackerManager(assoc_dist=100.0)
    initial = manager.update_tracks([_cluster(10_000.0, 1), _cluster(20_000.0, 2)], 1.0, reference)
    initial_ids = [track.track_id for track in initial]

    updated = manager.update_tracks([_cluster(20_100.0, 4), _cluster(10_100.0, 3)], 1.0, reference)
    positions_by_id = {track.track_id: float(np.asarray(track.state)[1]) for track in updated}

    assert set(positions_by_id) == set(initial_ids)
    assert positions_by_id[initial_ids[0]] < positions_by_id[initial_ids[1]]


def test_independent_same_tick_reports_update_one_existing_track():
    reference = Position(0.0, 0.0, 0.0)
    manager = TrackerManager(assoc_dist=100.0)
    initial = manager.update_tracks([_cluster(10_000.0, 1)], 1.0, reference)
    track_id = initial[0].track_id

    updated = manager.update_tracks(
        [_cluster(10_080.0, 2, 3), _cluster(10_120.0, 3, 4)],
        1.0,
        reference,
    )

    assert len(updated) == 1
    assert updated[0].track_id == track_id
    assert set(manager.track_meta[track_id]["source_ids"]) == {3, 4}
    assert set(manager.track_meta[track_id]["report_ids"]) == {2, 3}
    assert set(manager.track_meta[track_id]["report_lineage"]) == {(3, 1), (3, 2), (4, 3)}


def test_equal_local_report_ids_from_independent_sensors_are_not_deduplicated():
    reference = Position(0.0, 0.0, 0.0)
    manager = TrackerManager(assoc_dist=100.0)
    initial = manager.update_tracks([_cluster(10_000.0, 1, 3)], 1.0, reference)
    track_id = initial[0].track_id

    manager.update_tracks(
        [_cluster(10_080.0, 1, 3), _cluster(10_120.0, 1, 4)],
        1.0,
        reference,
    )

    assert set(manager.track_meta[track_id]["report_lineage"]) == {(3, 1), (4, 1)}


def test_classification_evidence_weights_lineage_and_sensor_independence():
    reference = Position(0.0, 0.0, 0.0)
    manager = TrackerManager(assoc_dist=100.0)
    belief = (0.8, 0.05, 0.05, 0.02, 0.08)
    first_cluster = _cluster(10_000.0, 1, 3)
    first_cluster["classification_probabilities"] = belief
    first = manager.update_tracks([first_cluster], 1.0, reference)[0]
    track_id = first.track_id
    initial_entropy = first.classification_entropy_nats

    manager.update_tracks([first_cluster], 1.0, reference)
    repeated_meta = manager.track_meta[track_id]
    assert repeated_meta["effective_classification_evidence"] == 1.0

    correlated = _cluster(10_050.0, 2, 3)
    correlated["classification_probabilities"] = belief
    correlated_snapshot = manager.update_tracks([correlated], 1.0, reference)[0]
    assert correlated_snapshot.effective_classification_evidence == 1.25

    independent = _cluster(10_100.0, 1, 4)
    independent["classification_probabilities"] = belief
    independent_snapshot = manager.update_tracks([independent], 1.0, reference)[0]
    assert independent_snapshot.effective_classification_evidence == 2.25
    correlated_drop = initial_entropy - correlated_snapshot.classification_entropy_nats
    independent_drop = (
        correlated_snapshot.classification_entropy_nats
        - independent_snapshot.classification_entropy_nats
    )
    assert independent_drop > correlated_drop


def test_production_cluster_never_needs_target_handle():
    manager = TrackerManager(assoc_dist=100.0)
    cluster = _cluster(5000.0, 1)
    assert "T" not in cluster
    output = manager.update_tracks([cluster], 1.0, Position(0.0, 0.0, 0.0))
    assert not hasattr(output[0], "target")
    assert not any(isinstance(value, SimpleNamespace) for value in manager.track_meta.values())


def test_track_lifecycle_confirms_coasts_and_reacquires_without_changing_id():
    manager = TrackerManager(assoc_dist=100.0)
    reference = Position(0.0, 0.0, 0.0)

    first = manager.update_tracks([_cluster(10_000.0, 1)], 1.0, reference)
    track_id = first[0].track_id
    assert manager.track_meta[track_id]["lifecycle"] is TrackLifecycle.TENTATIVE

    manager.update_tracks([_cluster(10_100.0, 2)], 1.0, reference)
    assert manager.track_meta[track_id]["lifecycle"] is TrackLifecycle.TENTATIVE

    manager.update_tracks([_cluster(10_200.0, 3)], 1.0, reference)
    assert manager.track_meta[track_id]["lifecycle"] is TrackLifecycle.CONFIRMED

    manager.update_tracks([], 1.0, reference)
    assert manager.track_meta[track_id]["lifecycle"] is TrackLifecycle.COASTING

    reacquired = manager.update_tracks([_cluster(10_400.0, 4)], 1.0, reference)
    assert reacquired[0].track_id == track_id
    assert manager.track_meta[track_id]["lifecycle"] is TrackLifecycle.REACQUIRED


def test_measurement_age_and_lifetime_are_independent():
    manager = TrackerManager(assoc_dist=100.0)
    reference = Position(0.0, 0.0, 0.0)

    for report_id in range(1, 6):
        fresh = manager.update_tracks(
            [_cluster(10_000.0 + 100.0 * report_id, report_id)], 1.0, reference
        )[0]

    assert fresh.age_s == 0.0
    mature_lifetime = fresh.lifetime_s
    assert mature_lifetime >= 5.0

    coasted = manager.update_tracks([], 1.0, reference)[0]
    assert coasted.age_s == 1.0
    assert coasted.lifetime_s == mature_lifetime + 1.0

    reacquired = manager.update_tracks([_cluster(10_700.0, 6)], 1.0, reference)[0]
    assert reacquired.age_s == 0.0
    assert reacquired.lifetime_s == coasted.lifetime_s + 1.0


def test_tentative_track_population_is_hard_capped():
    manager = TrackerManager(assoc_dist=100.0, max_tentative_tracks=3)
    clusters = [_cluster(10_000.0 + index * 10_000.0, index) for index in range(20)]

    manager.update_tracks(clusters, 1.0, Position(0.0, 0.0, 0.0))

    assert len(manager.tracks) == 3


def test_anonymous_association_skips_solves_for_impossible_pairs(monkeypatch):
    reference = Position(0.0, 0.0, 0.0)
    manager = TrackerManager(assoc_dist=100.0)
    manager.update_tracks([_cluster(10_000.0, 1)], 1.0, reference)

    def forbidden_solve(*_args, **_kwargs):
        raise AssertionError("distant association should fail the trace bound")

    monkeypatch.setattr(np.linalg, "solve", forbidden_solve)
    manager._associate_anonymous_clusters([_cluster(1_000_000.0, 2)], 1.0, reference)


def _count_measurement_conversions(manager, clusters, reference, monkeypatch):
    original = manager._cluster_to_track_measurement
    calls = 0

    def counted(cluster, ref):
        nonlocal calls
        calls += 1
        return original(cluster, ref)

    monkeypatch.setattr(manager, "_cluster_to_track_measurement", counted)
    manager._associate_anonymous_clusters(clusters, 1.0, reference)
    return calls


def test_anonymous_association_gating_conversions_scale_with_clusters(monkeypatch):
    # Association gates every track against every cluster, but converts each cluster
    # into the shared frame once (O(clusters)) rather than once per track-cluster pair
    # (O(tracks * clusters)); only the finalized associations re-convert a measurement
    # into their own track frame. Adding a track that gates to nothing must therefore
    # leave the conversion count unchanged, proving the cost is independent of the
    # non-associating track population.
    reference = Position(0.0, 0.0, 0.0)
    clusters = [_cluster(10_100.0, 3), _cluster(20_100.0, 4), _cluster(30_000.0, 5)]

    two_tracks = TrackerManager(assoc_dist=100.0)
    two_tracks.update_tracks([_cluster(10_000.0, 1), _cluster(20_000.0, 2)], 1.0, reference)
    calls_two = _count_measurement_conversions(two_tracks, clusters, reference, monkeypatch)

    # A third pre-existing track far from every incoming cluster gates to nothing.
    three_tracks = TrackerManager(assoc_dist=100.0)
    three_tracks.update_tracks(
        [_cluster(10_000.0, 1), _cluster(20_000.0, 2), _cluster(500_000.0, 6)], 1.0, reference
    )
    calls_three = _count_measurement_conversions(three_tracks, clusters, reference, monkeypatch)

    # 3 shared-frame gating conversions + 2 re-conversions for the two associations.
    assert calls_two == 5
    # The extra, non-associating track adds no gating conversions (O(clusters), not O(T*C)).
    assert calls_three == calls_two
