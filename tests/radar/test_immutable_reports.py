from types import SimpleNamespace

import numpy as np

from bvr_marl_core.radar.units.aircraft import AircraftRadar
from bvr_marl_core.simulator.core.helpers import Position


def test_cached_and_delayed_reports_do_not_retain_target_objects():
    owner = SimpleNamespace(
        id=7,
        group="blue",
        position=Position(48.0, 11.0, 5000.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        velocity=np.zeros(3),
        radar_emitting=True,
    )
    target = SimpleNamespace(
        id=9,
        group="red",
        position=Position(48.01, 11.0, 5000.0),
        velocity=np.zeros(3),
        orientation=0.0,
        rcs=10.0,
    )
    radar = AircraftRadar(
        horizontal_fov_deg=120.0,
        vertical_fov_deg=60.0,
        max_range_m=50_000.0,
        radar_frequency_hz=10e9,
        tx_power_w=1e10,
        antenna_gain_db=40.0,
        snr_threshold_db=-100.0,
        false_alarm_rate=0.0,
        range_resolution_m=100.0,
        angular_resolution_deg=1.0,
        dl_delay_base_s=1.0,
        owner=owner,
    )
    owner.radar = radar
    sim = SimpleNamespace(
        elapsed_time_s=2.0, ew_world=SimpleNamespace(collect_range_denial=lambda *_: {})
    )

    for _ in range(radar.scan_scheduler.sectors):
        radar.update(1.0, sim, [target], owner.position, group_radars=[])
        sim.elapsed_time_s += 1.0
        if radar.cached_detections:
            break
    report = radar.cached_detections[0]
    captured = tuple(report.measurement)
    target.position = Position(49.0, 12.0, 100.0)

    assert "T" not in report
    assert tuple(report.measurement) == captured
    assert radar.get_delayed_detections(100.0) == []
    assert radar.last_datalink_status == "rejected_before_history"
