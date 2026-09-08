from types import SimpleNamespace

import numpy as np

from bvr_marl_core.radar.core.data_link import DataLink
from bvr_marl_core.radar.core.track_reconcile import reconcile_local_and_network
from bvr_marl_core.radar.radar import Radar
from tests.helpers.track_snapshot import track_snapshot


def _track(track_id, east_m, variance, *, source, report_id):
    covariance = np.eye(6) * variance
    return track_snapshot(
        track_id,
        state=(east_m, 0.0, 0.0, 0.0, 0.0, 0.0),
        covariance=covariance,
        source_ids=(source,),
        report_lineage=((source, report_id),),
        reference=(0.0, 0.0, 0.0),
    )


def test_local_estimate_adopts_shared_track_number_and_combined_provenance():
    local = _track(7, 10_000.0, 25.0, source="local", report_id=4)
    network = _track(1_000_003, 10_020.0, 400.0, source="peer", report_id=8)

    result = reconcile_local_and_network([local], [network])

    assert len(result) == 1
    assert result[0].track_id == network.track_id
    assert result[0].state == local.state
    assert set(result[0].source_ids) == {"local", "peer"}
    assert set(result[0].report_lineage) == {("local", 4), ("peer", 8)}


def test_triangulated_network_estimate_beats_imprecise_local_strobe():
    local = _track(7, 60_000.0, 1_000_000_000.0, source="local", report_id=4)
    network = _track(1_000_003, 10_000.0, 100.0, source="peer", report_id=8)

    result = reconcile_local_and_network([local], [network])

    # The deliberately broad local covariance gates the two hypotheses, but the
    # triangulated network range is far more informative and is retained.
    assert result == [network]


def test_unmatched_local_and_network_contacts_are_both_preserved():
    local = _track(7, -100_000.0, 25.0, source="local", report_id=4)
    network = _track(1_000_003, 100_000.0, 25.0, source="peer", report_id=8)

    result = reconcile_local_and_network([local], [network])

    assert {track.track_id for track in result} == {7, 1_000_003}


def test_only_full_datalink_members_can_read_the_shared_picture():
    full = DataLink("full")
    own = DataLink("own")
    owner = SimpleNamespace(id=1, group="blue", radar=SimpleNamespace(data_link=full))
    peer = SimpleNamespace(id=2, group="blue", radar=SimpleNamespace(data_link=full))
    sim = SimpleNamespace(active_units={1: owner, 2: peer}, is_datalink_up=lambda *_: True)

    assert Radar._on_datalink_net(SimpleNamespace(), sim, owner) is True

    owner.radar.data_link = own
    assert Radar._on_datalink_net(SimpleNamespace(), sim, owner) is False


def test_no_inbound_link_forces_own_radar_fallback():
    full = DataLink("full")
    owner = SimpleNamespace(id=1, group="blue", radar=SimpleNamespace(data_link=full))
    peer = SimpleNamespace(id=2, group="blue", radar=SimpleNamespace(data_link=full))
    sim = SimpleNamespace(active_units={1: owner, 2: peer}, is_datalink_up=lambda *_: False)

    assert Radar._on_datalink_net(SimpleNamespace(), sim, owner) is False
