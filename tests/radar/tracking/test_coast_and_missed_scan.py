"""Coast + missed-scan benchmark mechanisms (paper imperfection axes)."""

import math
from types import SimpleNamespace

from bvr_marl_core.radar.tracking.helpers.track_manager import TRACK_DELETION_MISSED_UPDATES
from bvr_marl_core.radar.tracking.tracker import TrackerManager
from bvr_marl_core.simulator.core.helpers import Position

# ---- Coast horizon -------------------------------------------------------------


def _cluster(e, n, report_id=1):
    d = math.hypot(e, n)
    return {
        "az": math.degrees(math.atan2(e, n)),
        "el": 0.0,
        "d": d,
        "dop": 0.0,
        "T": None,
        "n_obs": 1,
        "is_false_alarm": False,
        "range_denied": False,
        "triangulated": False,
        "report_lineage": ((1, report_id),),
        "source_ids": (1,),
    }


def _confirm(tm, pos):
    for i in range(4):
        tm.update_tracks([_cluster(0.0, 30_000.0, report_id=i + 1)], dt=1.0, default_ref=pos)


def test_extra_coast_keeps_track_alive_longer():
    pos = Position(0, 0, 0)
    base = TrackerManager(assoc_dist=400.0, confirmation_hits=2)
    extended = TrackerManager(assoc_dist=400.0, confirmation_hits=2, coast_extra_updates=8)
    _confirm(base, pos)
    _confirm(extended, pos)

    # Coast both for exactly TRACK_DELETION_MISSED_UPDATES + a couple more scans: the
    # base tracker prunes, the extended one (bigger horizon) still holds the track.
    coast_steps = TRACK_DELETION_MISSED_UPDATES + 3
    for _ in range(coast_steps):
        base.update_tracks([], dt=1.0, default_ref=pos)
        extended.update_tracks([], dt=1.0, default_ref=pos)
    assert len(base.tracks) == 0
    assert len(extended.tracks) >= 1


def test_coast_extra_updates_defaults_to_zero():
    assert TrackerManager(assoc_dist=400.0).coast_extra_updates == 0


# ---- Missed scan ---------------------------------------------------------------


class _Streams:
    """Deterministic stream provider mimicking EpisodeRandomStreams."""

    def __init__(self, seed=0):
        import numpy as np

        self._rng = np.random.default_rng(seed)

    def generator(self, namespace, entity_id="episode"):
        return self._rng


def _radar_stub(p):
    # Reuse Radar._apply_missed_scan without constructing a full radar.
    from bvr_marl_core.radar.radar import Radar

    r = Radar.__new__(Radar)
    r.missed_scan_probability = p
    r.owner = SimpleNamespace(id=1)
    return r


def test_missed_scan_drops_fraction_of_detections():
    r = _radar_stub(0.5)
    sim = SimpleNamespace(random_streams=_Streams(0))
    dets = [{"az": i} for i in range(4000)]
    kept = r._apply_missed_scan(dets, sim)
    frac = len(kept) / len(dets)
    assert 0.45 < frac < 0.55  # ~half dropped


def test_missed_scan_zero_is_noop():
    r = _radar_stub(0.0)
    sim = SimpleNamespace(random_streams=_Streams(0))
    dets = [{"az": i} for i in range(100)]
    assert r._apply_missed_scan(dets, sim) is dets


def test_missed_scan_without_streams_keeps_all():
    r = _radar_stub(0.9)
    dets = [{"az": 1}, {"az": 2}]
    assert r._apply_missed_scan(dets, SimpleNamespace(random_streams=None)) == dets
