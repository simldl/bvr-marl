from types import SimpleNamespace

from bvr_marl_core.radar.core.network_picture import NETWORK_TRACK_ID_BASE
from bvr_marl_core.registry import get_aircraft_class
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator

_MAP = SimpleNamespace(
    left_lon=-5,
    right_lon=5,
    bottom_lat=-5,
    top_lat=5,
    min_alt=0,
    max_alt=20_000,
)


def _fighter(lat, lon, yaw, group):
    return get_aircraft_class("Eurofighter")(
        Position(lat, lon, 9_000.0), yaw, 250.0, group, _MAP, 0.0, 20_000.0
    )


def test_team_members_share_track_number_and_fall_back_when_cut_off():
    sim = Simulator(tick_secs=1.0, random_seed=7)
    left = _fighter(0.0, 0.0, 0.0, "BLUE")
    right = _fighter(0.0, 0.5, 0.0, "BLUE")
    jammer = _fighter(0.8, 0.25, 180.0, "RED")
    jammer.noise_jammer_burn_through_km = 20.0
    for unit in (left, right, jammer):
        sim.add_unit(unit)

    for _ in range(8):
        sim.do_tick()

    left_ids = {track.track_id for track in left.radar.cached_tracks}
    right_ids = {track.track_id for track in right.radar.cached_tracks}
    shared_ids = left_ids & right_ids
    assert shared_ids
    assert min(shared_ids) >= NETWORK_TRACK_ID_BASE
    shared_id = min(shared_ids)
    assert sim.evaluator_truth_id_for_contact(left.id, shared_id) == jammer.id
    assert sim.evaluator_truth_id_for_contact(right.id, shared_id) == jammer.id

    sim.datalink_drop_prob = 1.0
    sim.do_tick()

    assert all(track.track_id < NETWORK_TRACK_ID_BASE for track in left.radar.cached_tracks)
    assert all(track.track_id < NETWORK_TRACK_ID_BASE for track in right.radar.cached_tracks)
