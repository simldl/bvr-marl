"""End-to-end covariance preservation and age/lifetime separation (items 8, 9, 10).

Item 9: a covariance stays a valid 6x6 PSD matrix through snapshot -> tactical contact
        -> weapon track -> coast propagation -> body-frame rotation.
Item 8: measurement age (freshness) and lifetime (total existence) are separate fields,
        and coast grows freshness age while preserving the last-measurement time.
Item 10: oracle access is explicitly labelled (an oracle env config requires a reason).
"""

import numpy as np
import pytest

from bvr_marl_core.domain.information import TrackLifecycle, TrackSnapshot, WeaponTrack
from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.rl.environment.spaces.observation.helpers.covariance import (
    rotate_cov_to_body,
)


def _spd_cov(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((6, 6))
    return A @ A.T + np.eye(6) * 5.0  # symmetric positive-definite


def _snapshot(cov, *, state_time=100.0, last_meas=100.0, lifetime=40.0):
    return TrackSnapshot(
        track_id=7,
        state_time_s=state_time,
        state=(1000.0, 500.0, 200.0, -150.0, 10.0, 0.0),
        covariance=tuple(tuple(r) for r in cov),
        confidence=0.8,
        lifecycle=TrackLifecycle.CONFIRMED,
        last_measurement_time_s=last_meas,
        lifetime_s=lifetime,
    )


def _assert_valid_6x6(cov):
    P = np.asarray(cov, dtype=float)
    assert P.shape == (6, 6)
    assert np.allclose(P, P.T, atol=1e-6), "covariance must stay symmetric"
    assert np.min(np.linalg.eigvalsh(P)) > -1e-6, "covariance must stay PSD"


# ---- Item 9: covariance preservation end to end ---------------------------------


def test_snapshot_requires_6x6():
    with pytest.raises(ValueError):
        TrackSnapshot(
            track_id=1,
            state_time_s=0.0,
            state=(0,) * 6,
            covariance=tuple(tuple(r) for r in np.eye(3)),  # wrong shape
            confidence=0.5,
            lifecycle=TrackLifecycle.CONFIRMED,
        )


def test_covariance_preserved_snapshot_to_tactical_contact():
    cov = _spd_cov()
    snap = _snapshot(cov)
    contact = TacticalContact.from_track_snapshot(snap)
    assert np.allclose(np.asarray(contact.covariance), cov)
    _assert_valid_6x6(contact.covariance)


def test_covariance_grows_and_stays_valid_under_coast():
    cov = _spd_cov()
    snap = _snapshot(cov, state_time=100.0)
    coasted = snap.propagated(112.0)  # dt = 12 s
    _assert_valid_6x6(coasted.covariance)
    # Coasting a CV track must not shrink positional uncertainty.
    assert np.trace(np.asarray(coasted.covariance)[:3, :3]) >= np.trace(cov[:3, :3])
    assert coasted.lifecycle is TrackLifecycle.COASTING


def test_covariance_preserved_through_weapon_track():
    cov = _spd_cov()
    wt = WeaponTrack(_snapshot(cov, state_time=100.0), launch_time_s=100.0)
    est = wt.estimate_at(105.0)
    _assert_valid_6x6(est.covariance)
    assert est.track_id == 7  # identity preserved


def test_covariance_valid_after_body_frame_rotation():
    cov = _spd_cov()
    Pb = rotate_cov_to_body(cov, yaw_deg=37.0)
    _assert_valid_6x6(Pb)
    # An orthogonal rotation preserves total variance (trace).
    assert np.isclose(np.trace(Pb), np.trace(cov), rtol=1e-6)


# ---- Item 8: age (freshness) vs lifetime (total) --------------------------------


def test_age_and_lifetime_are_independent():
    # Measured 3 s ago but the track has existed for 40 s.
    snap = _snapshot(_spd_cov(), state_time=100.0, last_meas=97.0, lifetime=40.0)
    assert snap.age_s == pytest.approx(3.0)  # freshness: time since last measurement
    assert snap.lifetime_s == pytest.approx(40.0)  # total existence, unrelated
    assert snap.age_s != snap.lifetime_s


def test_coast_grows_freshness_age_and_preserves_last_measurement():
    snap = _snapshot(_spd_cov(), state_time=100.0, last_meas=100.0, lifetime=40.0)
    assert snap.age_s == pytest.approx(0.0)
    coasted = snap.propagated(108.0)  # 8 s of coast, no new measurement
    assert coasted.age_s == pytest.approx(8.0)  # freshness age grew with the coast
    assert coasted.last_measurement_time_s == pytest.approx(100.0)  # unchanged
    assert coasted.lifetime_s == pytest.approx(48.0)  # total existence grew too


# ---- Item 10: oracle access is explicitly labelled ------------------------------


def test_oracle_mode_requires_labelled_reason():
    from bvr_marl_core.rl.environment.gym.gym_components.config import BVREnvConfig

    # Sensor-limited is the default and needs no reason.
    assert BVREnvConfig.from_dict({}).information_mode == "sensor_limited"
    # Oracle must carry an explicit use-reason label.
    with pytest.raises(ValueError):
        BVREnvConfig.from_dict({"information_mode": "oracle"})
    cfg = BVREnvConfig.from_dict(
        {"information_mode": "oracle", "oracle_use_reason": "diagnostic upper bound"}
    )
    assert cfg.information_mode == "oracle"
    assert cfg.oracle_use_reason
