import pytest
import numpy as np
import torch
from radar.radar import Radar
from radar.obs.observation import RadarObsGenerator
from simulator.core.helpers import Position

class DummyLUT:
    freq_hz = 1e9
    def get_probability(self, dist, rcs): return 1.0

def make_target(lat=0.0, lon=0.0, alt=1000.0, rcs=1.0):
    T = type('Tgt', (), {})()
    T.position = type('Pos', (), {})()
    T.position.lat, T.position.lon, T.position.alt = lat, lon, alt
    T.rcs = rcs
    T.orientation = 0.0
    T.velocity = np.zeros(3)
    return T

@pytest.fixture
def obsgen():
    return RadarObsGenerator(
        horizontal_fov_deg=90, vertical_fov_deg=60, max_range_m=5000,
        lut=DummyLUT(), snr_threshold_db=0.0, false_alarm_rate=0.0,
        np_rng=np.random.default_rng(0), device=torch.device("cpu")
    )

def test_generate_one_detection(obsgen):
    own_pos = make_target()
    target = make_target(lat=0.01)
    dets = obsgen.generate(own_pos.position, [target], yaw_deg=0.0, pitch_deg=0.0)
    assert len(dets) == 1
    assert "az" in dets[0] and "el" in dets[0]

@pytest.mark.skip(reason="False alarms deprecated - now handled by ECM/jamming system in radar.ew module")
def test_generate_false_alarms(obsgen):
    # NOTE: False alarm generation has been replaced by the ECM/jamming system
    # See radar.ew.ecm_emitter and radar.ew.ew_world for the new implementation
    # This test is kept for reference but marked as skipped
    obsgen.false_alarm_rate = 5.0
    obsgen.np_rng = np.random.default_rng(123)
    dets = obsgen.generate(make_target().position, [], 0, 0)
    assert len(dets) > 0
    falses = [d for d in dets if d["T"] is None]
    assert all(isinstance(f["az"], float) for f in falses)


@pytest.fixture
def radar_position():
    return Position(lat=0.0, lon=0.0, alt=1_000.0)

@pytest.fixture
def deterministic_radar():
    return Radar(
        horizontal_fov_deg=60.0,
        vertical_fov_deg=30.0,
        max_range_m=10_000.0,
        radar_frequency_hz=10e9,
        tx_power_w=1e5,
        antenna_gain_db=30.0,
        snr_threshold_db=0.0,
        false_alarm_rate=0.0,
        range_resolution_m=50.0,
        angular_resolution_deg=2.0,
    )

def test_detection_boundaries(monkeypatch, deterministic_radar, radar_position):
    import torch
    from types import SimpleNamespace
    monkeypatch.setattr(torch, "rand", lambda *s, **kw: torch.zeros(*s, **kw))

    radar = deterministic_radar

    # 1. Innerhalb Reichweite und FOV
    target = SimpleNamespace(
        position=Position(
            lat=radar_position.lat + 0.005,
            lon=radar_position.lon,
            alt=radar_position.alt
        ),
        orientation=0.0, rcs=2.0, velocity=np.zeros(3)
    )
    dets = radar.obsgen.generate(radar_position, [target], yaw_deg=0.0, pitch_deg=0.0)
    assert len(dets) == 1

    # 2. Außerhalb Reichweite
    over_range_lat = radar_position.lat + (radar.max_range_m + 1) / 111_000.0
    target.position = Position(lat=over_range_lat, lon=radar_position.lon, alt=radar_position.alt)
    dets = radar.obsgen.generate(radar_position, [target], yaw_deg=0.0, pitch_deg=0.0)
    assert len(dets) == 0

    # 3. Außerhalb Azimut-FOV (rechts)
    angle = radar.h_fov_deg / 2
    dist = 5000.0
    az_rad = np.radians(angle + 0.1)
    x_east = np.sin(az_rad) * dist
    y_north = np.cos(az_rad) * dist
    delta_lat = y_north / 111_000
    delta_lon = x_east / (111_000 * np.cos(np.radians(radar_position.lat)))
    target.position = Position(
        lat=radar_position.lat + delta_lat,
        lon=radar_position.lon + delta_lon,
        alt=radar_position.alt
    )
    dets = radar.obsgen.generate(radar_position, [target], yaw_deg=0.0, pitch_deg=0.0)
    assert len(dets) == 0

    # 4. Genau am linken Rand (leicht innerhalb)
    angle = -radar.h_fov_deg / 2 + 0.1
    az_rad = np.radians(angle)
    x_east = np.sin(az_rad) * dist
    y_north = np.cos(az_rad) * dist
    delta_lat = y_north / 111_000
    delta_lon = x_east / (111_000 * np.cos(np.radians(radar_position.lat)))
    target.position = Position(
        lat=radar_position.lat + delta_lat,
        lon=radar_position.lon + delta_lon,
        alt=radar_position.alt
    )
    dets = radar.obsgen.generate(radar_position, [target], yaw_deg=0.0, pitch_deg=0.0)
    assert len(dets) == 1