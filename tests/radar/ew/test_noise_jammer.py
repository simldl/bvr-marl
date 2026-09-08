"""EW rework: noise jamming as range denial (Stage 1)."""

from types import SimpleNamespace

from bvr_marl_core.radar.ew.noise_jammer import burn_through_range_m
from bvr_marl_core.registry import get_aircraft_class
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator

_ML = SimpleNamespace(left_lon=-5, right_lon=5, bottom_lat=-5, top_lat=5, min_alt=0, max_alt=20000)


def _ef(lat, lon, yaw, group):
    return get_aircraft_class("Eurofighter")(
        Position(lat=lat, lon=lon, alt=9000.0), yaw, 250.0, group, _ML, 0.0, 20000.0
    )


def test_burn_through_zero_without_jammer():
    radar = SimpleNamespace(tx_power_w=18e3, antenna_gain_db=36.0)
    assert burn_through_range_m(radar, 0.0, 1.0) == 0.0


def test_burn_through_scales_with_radar_power_and_rcs():
    weak = SimpleNamespace(tx_power_w=18e3, antenna_gain_db=36.0)
    strong = SimpleNamespace(tx_power_w=25e3, antenna_gain_db=40.0)
    # A stronger radar burns through farther; a bigger RCS target too.
    assert burn_through_range_m(strong, 25.0, 1.0) > burn_through_range_m(weak, 25.0, 1.0)
    assert burn_through_range_m(weak, 25.0, 3.0) > burn_through_range_m(weak, 25.0, 1.0)


def test_reference_burn_through_matches_config():
    # Reference radar (18 kW, 36 dB) + sigma=1 -> burn-through == configured km.
    radar = SimpleNamespace(tx_power_w=18e3, antenna_gain_db=36.0)
    assert burn_through_range_m(radar, 25.0, 1.0) == 25_000.0


def _detect_once(range_km, jam_km):
    sim = Simulator(tick_secs=1.0)
    blue = _ef(0.0, 0.0, 0.0, "BLUE")
    red = _ef(range_km / 111.0, 0.0, 180.0, "RED")
    red.noise_jammer_burn_through_km = jam_km
    sim.add_unit(blue)
    sim.add_unit(red)
    for _ in range(3):
        sim.do_tick()
    dets = blue.radar.cached_detections or []
    return dets[0] if dets else None


def test_far_jammer_is_detected_but_range_denied():
    d = _detect_once(90, jam_km=25)
    assert d is not None, "jammer must still be detected (bearing strobe)"
    assert d.get("range_denied") is True
    # Range is denied -> reported at the strobe placeholder, not the true ~90 km.
    assert d["d"] > 150_000.0
    # Bearing is preserved for triangulation.
    assert "strobe_az" in d and "obs_lat" in d
    assert "jammer_id" not in d
    assert str(d["strobe_id"]).startswith(f"{d.source_id}:strobe:")


def test_close_jammer_inside_burn_through_keeps_range():
    d = _detect_once(10, jam_km=25)
    assert d is not None
    assert not d.get("range_denied", False)
    assert d["d"] < 30_000.0  # true short range recovered


# --- Stage 2: triangulation ------------------------------------------------

import numpy as np  # noqa: E402

from bvr_marl_core.radar.core.utils import geodetic_to_enu  # noqa: E402
from bvr_marl_core.radar.ew.triangulation import triangulate  # noqa: E402


def test_triangulate_needs_two_lines():
    assert triangulate([np.zeros(3)], [np.array([0.0, 1.0, 0.0])])[1] is False


def test_triangulate_rejects_parallel_bearings():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([100.0, 0.0, 0.0])
    u = np.array([0.0, 1.0, 0.0])  # both point due north -> no baseline angle
    assert triangulate([p1, p2], [u, u])[1] is False


def test_triangulate_recovers_known_point():
    target = np.array([5000.0, 40000.0, 0.0])
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([30000.0, 0.0, 0.0])
    u1 = (target - p1) / np.linalg.norm(target - p1)
    u2 = (target - p2) / np.linalg.norm(target - p2)
    pt, ok = triangulate([p1, p2], [u1, u2])
    assert ok
    assert np.linalg.norm(pt - target) < 1.0


def _jammer_track_error(two_radars):
    sim = Simulator(tick_secs=1.0)
    b1 = _ef(0.0, 0.0, 0.0, "BLUE")
    red = _ef(0.8, 0.25, 180.0, "RED")
    red.noise_jammer_burn_through_km = 20
    sim.add_unit(b1)
    if two_radars:
        sim.add_unit(_ef(0.0, 0.5, 0.0, "BLUE"))
    sim.add_unit(red)
    for _ in range(8):
        sim.do_tick()
    for t in b1.radar.cached_tracks or []:
        # Operational identity is a local anonymous strobe hypothesis.
        if str(t.emitter_hypothesis_id).startswith("strobe-hypothesis:"):
            true = np.array(
                geodetic_to_enu(
                    red.position.lat,
                    red.position.lon,
                    red.position.alt,
                    b1.position.lat,
                    b1.position.lon,
                    b1.position.alt,
                )
            )
            return np.linalg.norm(np.array(t.state[:3]) - true) / 1000.0
    return None


def test_triangulation_recovers_range_vs_bearing_only():
    lone = _jammer_track_error(two_radars=False)
    pair = _jammer_track_error(two_radars=True)
    assert lone is not None and pair is not None
    assert lone > 20.0  # one radar can't range the jammer
    assert pair < 6.0  # two datalinked radars triangulate it
    assert pair < lone / 3.0


# --- Stage 4: missiles vs jammer -------------------------------------------

from bvr_marl_core.radar.core.data_link import DataLink  # noqa: E402
from bvr_marl_core.registry import get_missile_class  # noqa: E402


class _Tgt:
    id = "jammer-1"


def _strobe(obs_lat, obs_lon, az):
    return {
        "range_denied": True,
        "strobe_id": f"local:{obs_lon}",
        "strobe_az": az,
        "strobe_el": 0.0,
        "obs_lat": obs_lat,
        "obs_lon": obs_lon,
        "obs_alt": 9000.0,
        "T": _Tgt(),
    }


def test_temporal_strobe_revisits_are_bounded_per_resolvable_bearing_cell():
    strobes = [
        {
            "source_id": 1,
            "frequency_band": "x",
            "strobe_az": 20.1,
            "strobe_el": 1.1,
            "acquisition_time_s": time_s,
            "report_id": report_id,
        }
        for report_id, time_s in enumerate((1.0, 2.0, 3.0), start=1)
    ]
    strobes.extend(
        [
            {**strobes[-1], "source_id": 2},
            {**strobes[-1], "strobe_az": 23.0, "report_id": 9},
        ]
    )

    bounded = DataLink._collapse_temporal_strobe_revisits(strobes)

    assert len(bounded) == 3
    assert {item["report_id"] for item in bounded if item["source_id"] == 1} == {3, 9}


def test_operational_strobe_fusion_ignores_evaluator_jammer_identity():
    own = Position(0.0, 0.0, 0.0)
    first = [_strobe(0.0, 0.0, 26.6), _strobe(0.0, 0.5, -26.6)]
    second = [dict(item, jammer_id=f"perturbed-{index}") for index, item in enumerate(first)]

    first_output = DataLink("full")._resolve_jammer_strobes(first, own, cooperative=True)
    second_output = DataLink("full")._resolve_jammer_strobes(second, own, cooperative=True)

    for left, right in zip(first_output, second_output, strict=True):
        assert left["measurement_position"] == right["measurement_position"]
        assert left["jammer_id"] == right["jammer_id"]


def test_cooperative_triangulates_but_noncooperative_is_bearing_only():
    dl = DataLink("full")
    own = Position(lat=0.0, lon=0.0, alt=9000.0)
    strobes = [_strobe(0.0, 0.0, 26.6), _strobe(0.0, 0.5, -26.6)]  # own + wingman

    coop = dl._resolve_jammer_strobes(strobes, own, cooperative=True)
    assert len(coop) == 1 and coop[0]["triangulated"] is True
    assert coop[0]["range_denied"] is False

    solo = dl._resolve_jammer_strobes(strobes, own, cooperative=False)
    # Only the own-platform bearing survives -> bearing-only home-on-jam.
    assert len(solo) == 1 and solo[0]["range_denied"] is True
    assert solo[0]["triangulated"] is False


def test_datalink_dropout_removes_links():
    sim = Simulator(tick_secs=1.0)
    a = _ef(0.0, 0.0, 0.0, "BLUE")
    b = _ef(0.0, 0.3, 0.0, "BLUE")
    c = _ef(0.0, -0.3, 0.0, "BLUE")
    for u in (a, b, c):
        sim.add_unit(u)

    sim.datalink_drop_prob = 0.0
    sim._sample_datalink_links()
    assert all(len(DataLink.update_group_radars(sim, owner=a)) == 2 for _ in range(20))

    sim.datalink_drop_prob = 1.0
    sim._sample_datalink_links()
    assert all(len(DataLink.update_group_radars(sim, owner=a)) == 0 for _ in range(20))

    sim.datalink_drop_prob = 0.5
    sizes = []
    for _ in range(400):
        sim._sample_datalink_links()
        first = len(DataLink.update_group_radars(sim, owner=a))
        # Every consumer sees the same already-sampled state during this tick.
        assert len(DataLink.update_group_radars(sim, owner=a)) == first
        sizes.append(first)
    assert 0.6 < (sum(sizes) / len(sizes)) < 1.4  # ~half of 2 links up


# --- Vectorized strobe association matches the scalar per-pair path ----------

import math  # noqa: E402

import pytest  # noqa: E402

from bvr_marl_core.radar.ew.triangulation import (  # noqa: E402
    pairwise_bearing_candidates,
    triangulate_pair_normalized,
)


def _scalar_pair_candidate(p_left, u_left, p_right, u_right):
    """Reference per-pair gate the vectorized kernel replaces: (accepted, residual)."""
    point, ok = triangulate_pair_normalized(p_left, u_left, p_right, u_right)
    if not ok:
        return False, math.inf
    residual = 0.0
    for observer, direction in ((p_left, u_left), (p_right, u_right)):
        offset = point - observer
        along = float(offset @ direction)
        if along <= 0.0:
            return False, math.inf
        residual += float(np.linalg.norm(offset - along * direction)) / along
    if residual <= 2.0 * math.sin(math.radians(5.0)):
        return True, residual
    return False, math.inf


def test_pairwise_bearing_candidates_matches_scalar_gate():
    rng = np.random.default_rng(2026)
    observers = rng.uniform(-50_000.0, 50_000.0, size=(24, 3))
    targets = rng.uniform(-120_000.0, 120_000.0, size=(24, 3))
    directions = targets - observers
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    residual, accept = pairwise_bearing_candidates(observers, directions)

    for left in range(len(observers)):
        for right in range(len(observers)):
            expected_accept, expected_residual = (
                _scalar_pair_candidate(
                    observers[left], directions[left], observers[right], directions[right]
                )
                if left < right
                else (False, math.inf)
            )
            assert bool(accept[left, right]) == expected_accept
            if expected_accept:
                assert residual[left, right] == pytest.approx(expected_residual, rel=1e-9)


def test_strobe_association_groups_two_receivers_on_one_emitter():
    # Two receivers with a wide baseline both see one emitter to the north; their
    # bearings intersect, so the strobes must land in a single triangulable group.
    own = Position(0.0, 0.0, 9000.0)
    both = DataLink("full")._associate_strobes_by_geometry(
        [_strobe(0.0, 0.0, 26.6), _strobe(0.0, 0.5, -26.6)], own, cooperative=True
    )
    assert len(both) == 1 and len(both[0]) == 2


def test_strobe_association_never_merges_a_single_observer():
    # Two bearings from the *same* observer cannot triangulate, so the observer mask
    # must keep them in separate groups no matter how well the geometry lines up.
    own = Position(0.0, 0.0, 9000.0)
    groups = DataLink("full")._associate_strobes_by_geometry(
        [_strobe(0.0, 0.0, 26.6), _strobe(0.0, 0.0, -26.6)], own, cooperative=True
    )
    assert sorted(len(group) for group in groups) == [1, 1]


def test_nextgen_missile_has_full_datalink_classic_does_not():
    ML = _ML
    shooter = _ef(0.0, 0.0, 0.0, "BLUE")
    tgt = _ef(0.5, 0.0, 180.0, "RED")
    jatm = get_missile_class("aim260")(0.0, tgt, shooter, ML)
    amraam = get_missile_class("amraam")(0.0, tgt, shooter, ML)
    assert jatm.full_datalink is True
    assert amraam.full_datalink is False
    # Missile seekers are jam-susceptible (so they can be denied range -> HOJ).
    assert jatm.radar.jam_susceptible is True
