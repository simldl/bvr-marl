"""Track-confidence signal spans the full [0, 1] range.

Regression guard for the fix that let a single-radar track reach ~1.0 confidence
instead of saturating near ~0.6 (``obs_score`` was previously normalised by the
maximum sensor count, so one radar could never contribute more than
``1 / N_MAX_RADARS`` to the detection term).  Confidence is a dense RL learning
signal, so a lone radar steadily holding an accurate, mature track must be able
to express near-full confidence.
"""

from __future__ import annotations

import numpy as np

from bvr_marl_core.radar.tracking.helpers.track_manager import calculate_confidence


class _FakeKF:
    def __init__(self, pos_std_m: float, missed_updates: float = 0.0) -> None:
        self.missed_updates = missed_updates
        self._cov = np.eye(6) * (pos_std_m**2)

    def get_covariance(self) -> np.ndarray:
        return self._cov


def _mature_single_radar_meta() -> dict:
    # One radar (obs_ema≈1), fully matured, measurements consistent (NIS at DOF).
    return {"obs_ema": 1.0, "updates_with_meas": 10, "nis_ema": 3.0}


def test_single_radar_track_reaches_high_confidence():
    conf = calculate_confidence(_mature_single_radar_meta(), _FakeKF(pos_std_m=100.0))
    assert conf > 0.9  # would have been capped near ~0.6 before the fix


def test_confidence_can_reach_unity_for_a_tight_track():
    conf = calculate_confidence(_mature_single_radar_meta(), _FakeKF(pos_std_m=1.0))
    assert conf > 0.99


def test_new_or_noisy_track_stays_low():
    # Freshly acquired (immature) and poorly localised → low confidence.
    meta = {"obs_ema": 1.0, "updates_with_meas": 1, "nis_ema": 3.0}
    conf = calculate_confidence(meta, _FakeKF(pos_std_m=3000.0))
    assert conf < 0.3


def test_missed_updates_and_inconsistency_reduce_confidence():
    base = calculate_confidence(_mature_single_radar_meta(), _FakeKF(pos_std_m=100.0))
    stale = calculate_confidence(
        _mature_single_radar_meta(), _FakeKF(pos_std_m=100.0, missed_updates=3.0)
    )
    inconsistent = calculate_confidence(
        {"obs_ema": 1.0, "updates_with_meas": 10, "nis_ema": 30.0},
        _FakeKF(pos_std_m=100.0),
    )
    assert stale < base
    assert inconsistent < base


def test_confidence_is_bounded():
    for pos_std in (0.5, 50.0, 5000.0):
        conf = calculate_confidence(_mature_single_radar_meta(), _FakeKF(pos_std_m=pos_std))
        assert 0.0 <= conf <= 1.0


class _FakeKFAtRange(_FakeKF):
    """A KF whose position estimate sits at a given range from the sensor."""

    def __init__(self, pos_std_m: float, range_m: float, missed_updates: float = 0.0) -> None:
        super().__init__(pos_std_m, missed_updates)
        self._state = np.array([range_m, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    def get_state(self) -> np.ndarray:
        return self._state


def test_range_adaptive_accuracy_keeps_bvr_tracks_confident():
    # A firm, mature track at 100 km whose position std (~2 km) is what the radar's
    # angular resolution allows at that range. A fixed 500 m accuracy reference
    # collapsed confidence to ~0 at BVR on large maps; the range-adaptive reference
    # judges *angular* quality, so this well-held track stays commit-worthy.
    meta = _mature_single_radar_meta()
    far = calculate_confidence(meta, _FakeKFAtRange(pos_std_m=2000.0, range_m=100_000.0))
    assert far > 0.6
    # The same absolute 2 km std at 5 km range IS poorly localised -> lower.
    close = calculate_confidence(meta, _FakeKFAtRange(pos_std_m=2000.0, range_m=5_000.0))
    assert close < far


def test_large_theater_confidence_crosses_doctrine_gates_after_confirmation():
    """Maintained angular-quality tracks remain actionable on 500 km maps."""
    for range_m in (100_000.0, 170_000.0, 300_000.0):
        pos_std_m = 0.02 * range_m
        confirmed = calculate_confidence(
            {"obs_ema": 1.0, "updates_with_meas": 3, "nis_ema": 3.0},
            _FakeKFAtRange(pos_std_m=pos_std_m, range_m=range_m),
        )
        weapons_quality = calculate_confidence(
            {"obs_ema": 1.0, "updates_with_meas": 4, "nis_ema": 3.0},
            _FakeKFAtRange(pos_std_m=pos_std_m, range_m=range_m),
        )
        assert confirmed >= 0.60
        assert weapons_quality >= 0.70


def test_large_absolute_error_is_not_hidden_by_large_map_range():
    good = calculate_confidence(
        _mature_single_radar_meta(),
        _FakeKFAtRange(pos_std_m=4000.0, range_m=200_000.0),
    )
    poor = calculate_confidence(
        _mature_single_radar_meta(),
        _FakeKFAtRange(pos_std_m=30_000.0, range_m=200_000.0),
    )
    assert good > poor
    assert poor < 0.6
