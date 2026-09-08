from datetime import timedelta
from types import SimpleNamespace

import numpy as np

from bvr_marl_core.aircraft.systems.sensor import AircraftSensorSystem
from bvr_marl_core.simulator.simulator import Simulator
from tests.helpers.track_snapshot import track_snapshot


def _sensor(observer_id=7):
    sensor = AircraftSensorSystem.__new__(AircraftSensorSystem)
    sensor.parent = SimpleNamespace(id=observer_id)
    sensor.sensor_tracks = []
    sensor.bda_confirmed = set()
    sensor.bda_probability = {}
    sensor._bda_history = {}
    sensor._bda_thresholds = {}
    return sensor


def _track(hidden_target, *, vertical_speed=-50.0, confidence=0.8):
    state = np.array([1_000.0, 2_000.0, 8_000.0, 150.0, 0.0, vertical_speed])
    covariance = np.eye(6) * 100.0
    return track_snapshot("track-12", state=state, covariance=covariance, confidence=confidence)


def test_bda_is_invariant_to_hidden_damage_state():
    healthy = SimpleNamespace(is_mortally_hit=False)
    damaged = SimpleNamespace(is_mortally_hit=True)
    sensor_a = _sensor()
    sensor_b = _sensor()
    sim_a = Simulator(random_seed=31)
    sim_b = Simulator(random_seed=31)

    for _ in range(5):
        sensor_a.sensor_tracks = [_track(healthy)]
        sensor_b.sensor_tracks = [_track(damaged)]
        sensor_a._update_bda(sim_a, 0.5)
        sensor_b._update_bda(sim_b, 0.5)
        sim_a.utc_time += timedelta(seconds=0.5)
        sim_b.utc_time += timedelta(seconds=0.5)

    assert sensor_a.bda_probability == sensor_b.bda_probability
    assert sensor_a.bda_confirmed == sensor_b.bda_confirmed


def test_persistent_observed_descent_raises_kill_assessment_probability():
    sensor = _sensor()
    sim = Simulator(random_seed=9)
    target = SimpleNamespace(is_mortally_hit=False)

    for _ in range(6):
        sensor.sensor_tracks = [_track(target, vertical_speed=-60.0)]
        sensor._update_bda(sim, 0.5)

    assert sensor.bda_probability["track-12"] > 0.75
