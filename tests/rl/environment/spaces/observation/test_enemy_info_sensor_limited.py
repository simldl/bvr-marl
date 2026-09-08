from types import SimpleNamespace

import numpy as np

from bvr_marl_core.rl.environment.spaces.observation.constants import d_EF
from bvr_marl_core.rl.environment.spaces.observation.enemy_info_builder import EnemyInfoBuilder
from tests.helpers.track_snapshot import track_snapshot


class HiddenTruth:
    """Evaluator object that fails the test if an operational field is read."""

    def __getattribute__(self, name):
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        raise AssertionError(f"sensor-limited observation read hidden truth field {name}")


class Config:
    information_mode = "sensor_limited"
    em_slots = 1
    ef_slots = 1


def _observation(hidden_truth):
    dlz = SimpleNamespace(
        r_min_m=2_000.0,
        r_tr_m=35_000.0,
        r_pi_m=55_000.0,
        r_aero_m=70_000.0,
        r_nez_in_m=2_000.0,
        r_nez_out_m=20_000.0,
    )
    estimate = SimpleNamespace(nominal=dlz, closing_speed_mps=350.0)
    track = track_snapshot(
        41,
        state=(10_000.0, 2_000.0, 500.0, -200.0, 0.0, 0.0),
        covariance=np.eye(6) * 100.0,
        confidence=0.75,
        classification="unknown",
    )
    sensor = SimpleNamespace(
        sensor_tracks=[track],
        get_locked_targets=lambda: set(),
        bda_confirmed=set(),
    )
    ownship = SimpleNamespace(
        id="blue",
        yaw_deg=0.0,
        sensor=sensor,
        wez=SimpleNamespace(
            compute_dlz_from_track=lambda state, covariance: estimate,
            zone_for_range=lambda distance, bands: "R2",
            sqi_from_estimate=lambda distance, closing, bands: 0.6,
        ),
    )
    simulator = SimpleNamespace(active_units={"blue": ownship})
    return EnemyInfoBuilder(simulator, Config()).build("blue")[1]


def test_sensor_limited_enemy_tokens_ignore_evaluator_target():
    first = _observation(HiddenTruth())
    second = _observation(HiddenTruth())
    assert first.shape == (1, d_EF)
    assert np.array_equal(first, second)
    assert first[0, -1] == 1.0
    # RID block remains unknown without an operational classification belief.
    assert np.array_equal(first[0, 13:17], np.zeros(4))
    # Estimated-state engagement aids remain available without evaluator geometry.
    assert first[0, 8] > 0.0
    assert first[0, 9] > 0.0
