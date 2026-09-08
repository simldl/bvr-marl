from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from bvr_marl_core.radar.core.friendly_picture import FriendlyPictureAdapter
from bvr_marl_core.rl.environment.spaces.observation.friendly_info_builder import (
    FriendlyInfoBuilder,
)
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.core.units import Velocity
from tests.helpers.track_snapshot import track_snapshot


def _unit(unit_id, group, lon, *, missile=False, track_id=None):
    return SimpleNamespace(
        id=unit_id,
        group=group,
        position=Position(0.0, lon, 8_000.0),
        velocity=Velocity(200.0, 0.0, 0.0),
        yaw_deg=0.0,
        is_missile=missile,
        phase="terminal" if missile else None,
        seeker_locked=missile,
        designated_track_id=track_id,
        data_link=(SimpleNamespace(get_mode=lambda: "full") if missile else None),
        sensor=SimpleNamespace(sensor_tracks=[]),
    )


def _config():
    return SimpleNamespace(
        information_mode="sensor_limited",
        fm_slots=2,
        ff_slots=2,
        ef_slots=2,
        all_agent_ids=("blue-1", "blue-2"),
    )


def test_sensor_limited_friendly_tokens_use_link_gated_immutable_reports():
    receiver = _unit("blue-1", "blue", 0.0)
    wingman = _unit("blue-2", "blue", 0.01)
    missile = _unit(30, "blue", 0.02, missile=True, track_id=71)
    enemy = _unit("red-1", "red", 0.03)
    receiver.sensor.sensor_tracks = [track_snapshot(71, state=np.zeros(6), reference=None)]
    sim = SimpleNamespace(
        elapsed_time_s=4.0,
        active_units={u.id: u for u in (receiver, wingman, missile, enemy)},
        is_datalink_up=lambda sender_id, receiver_id: sender_id != "blue-2",
    )

    missiles, fighters = FriendlyInfoBuilder(sim, _config()).build(receiver.id)

    assert missiles[0, -1] == pytest.approx(1.0)
    assert missiles[0, 6] == pytest.approx(1.0)
    assert missiles[0, 7] == pytest.approx(1.0)
    assert missiles[0, 8] == pytest.approx(1.0)
    assert missiles[0, 9] == pytest.approx(0.0)
    assert np.count_nonzero(fighters) == 0


def test_friendly_report_is_frozen_and_detached_from_source_mutation():
    receiver = _unit(1, "blue", 0.0)
    wingman = _unit(2, "blue", 0.01)
    sim = SimpleNamespace(elapsed_time_s=2.0, active_units={1: receiver, 2: wingman})
    report = FriendlyPictureAdapter(sim).reports_for(receiver)[0]

    original_state = report.relative_state_enu
    wingman.position.lon = 0.5

    assert report.relative_state_enu == original_state
    with pytest.raises(FrozenInstanceError):
        report.age_s = 3.0
