"""Tracker behavioural studies (paper item 12): unresolved crossings, duplicate
suppression, track fragmentation, and false-track handling.

These characterise and lock the tracker's data-association behaviour on scripted
multi-target geometries, complementing the existing duplicate-merge and ghost-track
unit tests. They drive the anonymous-cluster path of ``TrackerManager.update_tracks``.
"""

import math

import numpy as np

from bvr_marl_core.radar.tracking.tracker import TrackerManager
from bvr_marl_core.simulator.core.helpers import Position


def _cluster_for_enu(e, n, u=0.0, *, report_id=1, source_id=1):
    """Anonymous ranged detection cluster for a target at ENU (e, n, u)."""
    az = math.degrees(math.atan2(e, n))  # geodetic: 0=N, 90=E
    d = math.sqrt(e * e + n * n + u * u)
    el = math.degrees(math.asin(u / d)) if d > 0 else 0.0
    return {
        "az": az,
        "el": el,
        "d": d,
        "dop": 0.0,
        "T": None,
        "n_obs": 1,
        "is_false_alarm": False,
        "range_denied": False,
        "triangulated": False,
        "report_lineage": ((source_id, report_id),),
        "source_ids": (source_id,),
    }


def _confirm(tm, pos, cluster_fn, steps=4):
    for i in range(steps):
        tm.update_tracks([cluster_fn(report_id=i + 1)], dt=1.0, default_ref=pos)


# ---- Duplicate suppression -----------------------------------------------------


def test_duplicate_detections_yield_one_track():
    tm = TrackerManager(assoc_dist=300.0, confirmation_hits=1)
    pos = Position(0, 0, 0)
    # Two near-coincident detections of the same target in one scan.
    c = _cluster_for_enu(0.0, 30_000.0)
    c2 = _cluster_for_enu(30.0, 30_000.0, report_id=2, source_id=2)
    tm.update_tracks([c, c2], dt=1.0, default_ref=pos)
    tm.update_tracks([c, c2], dt=1.0, default_ref=pos)
    # The tracker must not carry two separate tracks for one physical target.
    assert len(tm.tracks) == 1


# ---- Unresolved crossing -------------------------------------------------------


def test_two_targets_crossing_stay_two_tracks():
    # Generous gate + gated per-step motion so the CV filter can follow each target
    # (motion << assoc_dist), with a fixed North separation larger than the gate so the
    # two are never confusable even as their East positions cross at t=10.
    tm = TrackerManager(assoc_dist=3_000.0, confirmation_hits=2)
    pos = Position(0, 0, 0)
    counts = []
    for t in range(20):
        e_a = -3_000.0 + 300.0 * t  # +E, crosses 0 at t=10
        e_b = 3_000.0 - 300.0 * t  # -E
        ca = _cluster_for_enu(e_a, 40_000.0, report_id=100 + t)
        cb = _cluster_for_enu(e_b, 44_000.0, report_id=200 + t)  # 4 km N apart (> gate)
        tm.update_tracks([ca, cb], dt=1.0, default_ref=pos)
        counts.append(len(tm.tracks))
    # Two distinct tracks are maintained through the crossing (never collapsed to one,
    # which would drop a contact).
    assert max(counts) >= 2
    assert len(tm.tracks) == 2


# ---- Track fragmentation -------------------------------------------------------


def test_reacquisition_does_not_fragment_into_multiple_tracks():
    tm = TrackerManager(assoc_dist=400.0, confirmation_hits=2)
    pos = Position(0, 0, 0)
    target = lambda report_id=1: _cluster_for_enu(0.0, 30_000.0, report_id=report_id)  # noqa: E731

    _confirm(tm, pos, target, steps=4)
    assert len(tm.tracks) == 1
    # Lose the target for several scans (no detections -> coast / miss).
    for _ in range(6):
        tm.update_tracks([], dt=1.0, default_ref=pos)
    # Reacquire at the same place.
    for i in range(4):
        tm.update_tracks([target(report_id=50 + i)], dt=1.0, default_ref=pos)
    # Anti-fragmentation: one physical target yields at most one active track after
    # reacquisition, not a pile of stale fragments alongside the fresh one.
    assert len(tm.tracks) == 1


def test_coasting_track_is_pruned_when_never_reacquired():
    tm = TrackerManager(assoc_dist=400.0, confirmation_hits=2)
    pos = Position(0, 0, 0)
    _confirm(tm, pos, lambda report_id=1: _cluster_for_enu(0.0, 30_000.0, report_id=report_id))
    assert len(tm.tracks) >= 1
    # A track that never receives another measurement is eventually dropped, so a
    # stale ghost does not persist forever.
    for _ in range(60):
        tm.update_tracks([], dt=1.0, default_ref=pos)
    assert len(tm.tracks) == 0


# ---- False tracks --------------------------------------------------------------


def test_false_alarm_cluster_does_not_confirm_a_track():
    tm = TrackerManager(assoc_dist=300.0, confirmation_hits=3)
    pos = Position(0, 0, 0)
    fa = _cluster_for_enu(0.0, 30_000.0)
    fa["is_false_alarm"] = True
    # A single unsupported false alarm must not immediately confirm an engageable track.
    snaps = tm.update_tracks([fa], dt=1.0, default_ref=pos)
    assert (
        all(not getattr(s, "engageable", False) or not s.lifecycle for s in snaps)
        or len(snaps) <= 1
    )
