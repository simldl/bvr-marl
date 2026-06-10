"""Tests for the range-scaled datalink staleness (delayed track sharing)."""

from collections import deque
from types import SimpleNamespace

from bvr_marl_core.radar.core.data_link import DataLink
from bvr_marl_core.radar.radar import Radar


def _radar(**kw):
    return Radar(
        horizontal_fov_deg=60,
        vertical_fov_deg=30,
        max_range_m=100_000,
        radar_frequency_hz=10e9,
        tx_power_w=15_000,
        antenna_gain_db=30,
        snr_threshold_db=8,
        **kw,
    )


def test_get_delayed_detections_returns_aged_snapshot():
    r = _radar(dl_delay_base_s=2.0)
    r._dl_history = deque([(0.0, ["t0"]), (1.0, ["t1"]), (2.0, ["t2"]), (3.0, ["t3"])])
    r.cached_detections = ["t3"]
    assert r.get_delayed_detections(0.0) == ["t3"]  # no delay -> live
    assert r.get_delayed_detections(2.0) == ["t1"]  # latest 3.0 - 2.0 = 1.0
    assert r.get_delayed_detections(10.0) == ["t0"]  # beyond history -> oldest retained


def test_get_delayed_detections_no_history_returns_current():
    r = _radar(dl_delay_base_s=2.0)
    r.cached_detections = ["cur"]
    assert r.get_delayed_detections(2.0) == ["cur"]


def test_dl_delay_disabled_by_default():
    r = _radar()
    assert r._dl_delay_enabled is False
    assert r.dl_delay_base_s == 0.0 and r.dl_delay_per_km_s == 0.0


def test_shared_source_detections_applies_delay_for_delayed_source():
    dl = DataLink("full")

    class Src:
        dl_delay_base_s = 1.5
        dl_delay_per_km_s = 0.0
        cached_detections = ["live"]

        def get_delayed_detections(self, d):
            return [f"delayed_{d:.1f}"]

    pos = SimpleNamespace(lat=0.0, lon=0.0, alt=0.0)  # same point -> range 0 -> delay = base
    assert dl._shared_source_detections(Src(), pos, pos) == ["delayed_1.5"]


def test_shared_source_detections_live_for_instant_source():
    dl = DataLink("full")

    class Src:
        dl_delay_base_s = 0.0
        dl_delay_per_km_s = 0.0
        cached_detections = ["live"]

    pos = SimpleNamespace(lat=0.0, lon=0.0, alt=0.0)
    assert dl._shared_source_detections(Src(), pos, pos) == ["live"]
