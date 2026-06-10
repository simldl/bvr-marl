import pytest

from bvr_marl_core.radar.core.data_link import DataLink


class DummyTarget:
    def __init__(self, id):
        self.id = id


class DummyClusterer:
    def cluster(self, detections):
        return detections  # returns detections as "clusters" directly


def make_detection(id_str, az=0.0, el=0.0, d=1000.0):
    return {
        "lat": 1,
        "lon": 2,
        "alt": 3,
        "az": az,
        "el": el,
        "d": d,
        "dop": 0.0,
        "T": [DummyTarget(id_str)],
    }


def make_dummy_radar(detections, mode):
    radar = type("DummyRadar", (), {})()
    radar.cached_detections = detections
    radar.data_link = DataLink(mode)
    # provide minimal owner the implementation expects
    radar.owner = type("Owner", (), {"id": 1})()
    return radar


def make_pos(lat=0, lon=0, alt=0):
    return type("Pos", (), dict(lat=lat, lon=lon, alt=alt))()


def test_modes_are_valid():
    for mode in DataLink.VALID_MODES:
        d = DataLink(mode)
        assert d.get_mode() == mode


def test_mode_invalid_raises():
    with pytest.raises(ValueError):
        DataLink("invalid")


def test_get_fused_clusters_own_and_full():
    dummy_radar = make_dummy_radar([make_detection("own")], mode="own")
    dummy_radar2 = make_dummy_radar([make_detection("other")], mode="full")
    own_pos = make_pos()
    dummy_clusterer = DummyClusterer()

    # own-Mode: only own detections
    d = DataLink("own")
    res = d.get_fused_clusters(
        dummy_radar, [(dummy_radar2, own_pos)], own_pos, clusterer=dummy_clusterer
    )
    ids = [c.get("T").id if c.get("T") else None for c in res]
    assert "own" in ids
    assert "other" not in ids

    # full-Mode: own + those from others that are in full mode
    d = DataLink("full")
    res = d.get_fused_clusters(
        dummy_radar, [(dummy_radar2, own_pos)], own_pos, clusterer=dummy_clusterer
    )
    ids = [c.get("T").id if c.get("T") else None for c in res]
    assert "own" in ids
    assert "other" in ids


def test_get_fused_clusters_other():
    dummy_radar = make_dummy_radar([make_detection("own")], mode="other")
    dummy_radar2 = make_dummy_radar([make_detection("other")], mode="own")
    own_pos = make_pos()
    dummy_clusterer = DummyClusterer()

    d = DataLink("other")
    res = d.get_fused_clusters(
        dummy_radar, [(dummy_radar2, own_pos)], own_pos, clusterer=dummy_clusterer
    )
    ids = [c.get("T").id if c.get("T") else None for c in res]
    assert "other" in ids
    assert "own" not in ids


def test_get_fused_clusters_none():
    dummy_radar = make_dummy_radar(
        [{"lat": 1, "lon": 2, "alt": 3, "az": 0.0, "el": 0.0, "d": 1000.0, "dop": 0.0, "T": []}],
        mode="none",
    )
    own_pos = make_pos()
    dummy_clusterer = DummyClusterer()

    d = DataLink("none")
    res = d.get_fused_clusters(dummy_radar, [], own_pos, clusterer=dummy_clusterer)
    assert res == []


def test_repr_contains_mode():
    d = DataLink("own")
    assert "own" in str(d)
