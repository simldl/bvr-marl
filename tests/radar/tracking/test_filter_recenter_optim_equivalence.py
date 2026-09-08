"""Numerical-equivalence guards for the ENU covariance-block rotation (and the
CV-filter predict, which must keep matching the explicit F@x / F@P@F.T form).

The covariance rotation was re-expressed as ``S @ cov @ S.T`` with
``S = blkdiag(R, R)``; it must match the per-3x3-block reference to far tighter
than the 1% tolerance required for the tracking/recentering math.
"""

import numpy as np

from bvr_marl_core.radar.core.utils import geodetic_to_enu
from bvr_marl_core.radar.tracking.filter.constant_velocity_filter import (
    ConstantVelocityKFFilter,
)
from bvr_marl_core.radar.tracking.helpers.enu_utils import enu_rotation
from bvr_marl_core.radar.tracking.helpers.recenter_logic import (
    _rotate_covariance_blocks,
    transform_state_between_refs,
)

# Far stricter than the 1% requirement — these are exact rewrites.
TOL = 1e-9


def _max_rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(np.abs(b), 1.0)  # absolute floor avoids div-by-tiny blowups
    return float(np.max(np.abs(a - b) / denom))


def _random_spd(rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(6, 6))
    return a @ a.T + 6.0 * np.eye(6)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    return enu_rotation(
        rng.uniform(-70, 70),
        rng.uniform(-180, 180),
        rng.uniform(-70, 70),
        rng.uniform(-180, 180),
    )


def _ref_rotate_cov(cov6: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Original 8-matmul block rotation (reference)."""
    Rt = R.T
    out = np.empty((6, 6), dtype=float)
    out[:3, :3] = R @ cov6[:3, :3] @ Rt
    out[:3, 3:] = R @ cov6[:3, 3:] @ Rt
    out[3:, :3] = R @ cov6[3:, :3] @ Rt
    out[3:, 3:] = R @ cov6[3:, 3:] @ Rt
    return out


def test_rotate_covariance_blocks_matches_reference():
    rng = np.random.default_rng(20260617)
    worst = 0.0
    for _ in range(500):
        P = _random_spd(rng)
        R = _random_rotation(rng)
        worst = max(worst, _max_rel_err(_rotate_covariance_blocks(P, R), _ref_rotate_cov(P, R)))
    assert worst < TOL, f"covariance rotation rel err {worst:.2e} exceeds {TOL:.0e}"


def test_predict_matches_F_matmul_reference():
    """Block-algebra predict must equal the explicit F@x / F@P@F.T + Q form."""
    rng = np.random.default_rng(7)
    worst_x = 0.0
    worst_P = 0.0
    for _ in range(500):
        meas = rng.normal(scale=1000.0, size=3)
        kf = ConstantVelocityKFFilter(meas, dt=1.0, process_noise_std=rng.uniform(1.0, 30.0))
        x0 = rng.normal(scale=500.0, size=6)
        P0 = _random_spd(rng)
        kf.set_state(x0, P0)

        dt = float(rng.uniform(0.1, 2.0))
        # Build the reference F/Q for this dt via the filter's own matrices.
        kf.dt = max(1e-6, dt)
        kf._apply_dt()
        F = kf.F.copy()
        Q = kf.Q.copy()
        ref_x = F @ x0
        ref_P = F @ P0 @ F.T + Q

        kf.predict(dt)
        worst_x = max(worst_x, _max_rel_err(kf.get_state(), ref_x))
        worst_P = max(worst_P, _max_rel_err(kf.get_covariance(), ref_P))

    assert worst_x < TOL, f"predict state rel err {worst_x:.2e} exceeds {TOL:.0e}"
    assert worst_P < TOL, f"predict cov rel err {worst_P:.2e} exceeds {TOL:.0e}"


def test_transform_state_covariance_round_trips():
    """A from->to->from transform must return the original state+cov (uses the
    optimized covariance rotation under the hood)."""
    from bvr_marl_core.simulator.core.helpers import Position

    rng = np.random.default_rng(99)
    worst = 0.0
    for _ in range(200):
        ref_a = Position(rng.uniform(-60, 60), rng.uniform(-180, 180), rng.uniform(0, 12000))
        ref_b = Position(
            ref_a.lat + rng.uniform(-0.5, 0.5),
            ref_a.lon + rng.uniform(-0.5, 0.5),
            ref_a.alt + rng.uniform(-2000, 2000),
        )
        x = rng.normal(scale=500.0, size=6)
        P = _random_spd(rng)
        x_b, P_b = transform_state_between_refs(x, P, ref_a, ref_b)
        x_back, P_back = transform_state_between_refs(x_b, P_b, ref_b, ref_a)
        worst = max(worst, _max_rel_err(x_back, x), _max_rel_err(P_back, P))
    assert worst < 1e-6, f"round-trip rel err {worst:.2e} exceeds 1e-6"
