from types import SimpleNamespace

import numpy as np

from bvr_marl_core.domain.information import FrameReference, SensorReport, SensorType
from bvr_marl_core.radar.core.data_link import DataLink
from bvr_marl_core.radar.core.network_picture import (
    NETWORK_TRACK_ID_BASE,
    NetworkTrackPicture,
    _transmitting_members,
)
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator


def _report(source_id, report_id, *, east_m=10_000.0, time_s=0.0):
    return SensorReport(
        report_id=report_id,
        source_id=source_id,
        acquisition_time_s=time_s,
        measurement=(90.0, 0.0, east_m),
        covariance=np.diag([0.05**2, 0.05**2, 50.0**2]),
        frame=FrameReference(0.0, 0.0, 0.0),
        sensor_type=SensorType.RADAR,
    )


def test_network_picture_fuses_sources_once_under_shared_track_number():
    reference = Position(0.0, 0.0, 0.0)
    picture = NetworkTrackPicture("blue", reference, assoc_dist=5_000.0, confirmation_hits=1)

    tracks = picture.update(
        [_report("radar-a", 1), _report("radar-b", 1, east_m=10_020.0)],
        tick_secs=1.0,
        current_time_s=0.0,
    )

    assert len(tracks) == 1
    assert tracks[0].track_id >= NETWORK_TRACK_ID_BASE
    assert set(tracks[0].source_ids) == {"radar-a", "radar-b"}
    assert set(tracks[0].report_lineage) == {("radar-a", 1), ("radar-b", 1)}

    projected = picture.tracks_in_frame(Position(0.0, 0.05, 1_000.0))
    assert len(projected) == 1
    assert projected[0].track_id == tracks[0].track_id
    assert projected[0].reference_frame == FrameReference(0.0, 0.05, 1_000.0)
    assert np.all(np.isfinite(projected[0].state))
    assert projected[0] is not tracks[0]
    assert tracks[0].reference_frame == FrameReference(0.0, 0.0, 0.0)
    assert projected[0].source_ids == tracks[0].source_ids
    assert projected[0].report_lineage == tracks[0].report_lineage


def test_net_entry_latency_is_source_based_not_receiver_distance():
    marker = _report("source", 1)
    radar = SimpleNamespace(
        dl_delay_base_s=1.5,
        dl_delay_per_km_s=0.25,
        get_delayed_detections=lambda delay_s: [marker] if delay_s == 1.5 else [],
    )
    source_position = Position(45.0, 2.0, 8_000.0)

    assert DataLink("full").net_entry_reports(radar, source_position) == [marker]


def test_only_members_with_an_outgoing_link_contribute_to_the_net():
    members = [SimpleNamespace(id=value) for value in (1, 2, 3)]
    up = {(1, 2), (2, 1)}
    sim = SimpleNamespace(is_datalink_up=lambda sender, receiver: (sender, receiver) in up)

    assert [member.id for member in _transmitting_members(sim, members)] == [1, 2]


def test_simulator_reset_discards_persistent_network_track_state():
    sim = Simulator()
    sim.network_pictures["blue"] = object()

    sim.reset_sim({})

    assert sim.network_pictures == {}
