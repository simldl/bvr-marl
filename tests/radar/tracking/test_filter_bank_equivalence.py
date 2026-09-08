"""Differential equivalence harness: vectorized CVFilterBank vs per-object filter.

The bank (filter_bank.py) is a numerical twin of ConstantVelocityKFFilter. This
suite pins the two together: identical inputs must yield identical state /
covariance / NIS to far tighter than the 1% the tracking math actually needs.

It exercises the correctness-critical paths called out in the design risk map:
  R2  long-horizon drift (1000-step track) — not just single-step
  R4  coast tracks (predict, no update) masked out of the batched update
  R7  singular innovation covariance -> K=0, NIS=1e6 fallback

If any assertion trips, the batched path is wrong: fail closed, do not ship.
"""

import numpy as np

from bvr_marl_core.radar.tracking.filter.constant_velocity_filter import (
    ConstantVelocityKFFilter,
    _inv3_symmetric,
)
from bvr_marl_core.radar.tracking.filter.filter_bank import (
    CVFilterBank,
    _inv3_symmetric_batch,
)

# Same bar the repo uses for "this is an exact rewrite" (see
# test_filter_recenter_optim_equivalence.py): rel err with an absolute floor of 1.
TOL = 1e-9


def _max_rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(np.abs(b), 1.0)
    return float(np.max(np.abs(a - b) / denom))


def _random_spd(rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(6, 6))
    return a @ a.T + 6.0 * np.eye(6)


def _make_pair(rng, M):
    """Build M per-object filters and one bank seeded from identical state."""
    x0 = rng.normal(scale=500.0, size=(M, 6))
    P0 = np.stack([_random_spd(rng) for _ in range(M)])
    sigma_a = rng.uniform(1.0, 50.0, size=M)

    objs = []
    for j in range(M):
        kf = ConstantVelocityKFFilter(x0[j, :3], dt=1.0, process_noise_std=float(sigma_a[j]))
        kf.set_state(x0[j], P0[j])
        objs.append(kf)

    bank = CVFilterBank(x0.copy(), P0.copy(), sigma_a.copy(), R_var=np.ones((M, 3)))
    return objs, bank


def _compare(objs, bank, idx=None) -> float:
    """Max rel err of state+cov across all tracks (or a subset of indices)."""
    worst = 0.0
    rng_idx = range(bank.size) if idx is None else idx
    for j in rng_idx:
        worst = max(
            worst,
            _max_rel_err(bank.get_state()[j], objs[j].get_state()),
            _max_rel_err(bank.get_covariance()[j], objs[j].get_covariance()),
        )
    return worst


# ---------------------------------------------------------------------------
# R7: batched 3x3 inverse must match the scalar closed form, including the
# singular fallback (scalar returns None; batch flags valid=False).
# ---------------------------------------------------------------------------
def test_inv3_batch_matches_scalar_and_flags_singular():
    rng = np.random.default_rng(11)
    mats = []
    expect_valid = []
    for _ in range(400):
        A = rng.normal(size=(3, 3))
        S = A @ A.T + 0.5 * np.eye(3)  # SPD, well-conditioned
        mats.append(S)
        expect_valid.append(True)
    # Inject exactly-singular matrices (rank-deficient).
    for _ in range(50):
        v = rng.normal(size=(3, 1))
        mats.append(v @ v.T)  # rank 1 -> det 0
        expect_valid.append(False)

    S_stack = np.stack(mats)
    inv_batch, valid = _inv3_symmetric_batch(S_stack)

    worst = 0.0
    for k, S in enumerate(mats):
        scalar = _inv3_symmetric(S)
        if scalar is None:
            assert not valid[k], f"row {k}: scalar singular but batch marked valid"
        else:
            assert valid[k], f"row {k}: scalar invertible but batch marked singular"
            worst = max(worst, _max_rel_err(inv_batch[k], scalar))
    assert worst < TOL, f"batched 3x3 inverse rel err {worst:.2e} exceeds {TOL:.0e}"


# ---------------------------------------------------------------------------
# Predict-only equivalence with heterogeneous per-track dt.
# ---------------------------------------------------------------------------
def test_predict_batch_matches_per_object():
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(50):
        M = int(rng.integers(1, 12))
        objs, bank = _make_pair(rng, M)
        dt = rng.uniform(0.05, 2.0, size=M)
        for j in range(M):
            objs[j].predict(float(dt[j]))
        bank.predict(dt)
        worst = max(worst, _compare(objs, bank))
    assert worst < TOL, f"predict rel err {worst:.2e} exceeds {TOL:.0e}"


# ---------------------------------------------------------------------------
# Single predict+update step with per-track measurement std (anisotropic R).
# ---------------------------------------------------------------------------
def test_update_batch_matches_per_object():
    rng = np.random.default_rng(3)
    worst = 0.0
    worst_nis = 0.0
    for _ in range(50):
        M = int(rng.integers(1, 12))
        objs, bank = _make_pair(rng, M)
        dt = rng.uniform(0.05, 2.0, size=M)
        std = rng.uniform(2.0, 200.0, size=(M, 3))
        z = bank.get_state()[:, :3] + rng.normal(scale=50.0, size=(M, 3))

        for j in range(M):
            objs[j].predict(float(dt[j]))
            objs[j].set_measurement_std(tuple(std[j]))
            objs[j].update(z[j])
        bank.predict(dt)
        bank.set_measurement_std(std)
        bank.update(z)

        worst = max(worst, _compare(objs, bank))
        for j in range(M):
            worst_nis = max(
                worst_nis,
                abs(bank.last_nis[j] - objs[j].get_last_update_stats()["nis"])
                / max(1.0, abs(objs[j].get_last_update_stats()["nis"])),
            )
    assert worst < TOL, f"update state/cov rel err {worst:.2e} exceeds {TOL:.0e}"
    assert worst_nis < TOL, f"update NIS rel err {worst_nis:.2e} exceeds {TOL:.0e}"


# ---------------------------------------------------------------------------
# R2 + R4: long-horizon track with coast gaps (predict, no update) masked out.
# ---------------------------------------------------------------------------
def test_long_horizon_with_coast_masking():
    rng = np.random.default_rng(4)
    M = 8
    objs, bank = _make_pair(rng, M)

    for step in range(1000):
        dt = rng.uniform(0.05, 1.0, size=M)
        std = rng.uniform(2.0, 200.0, size=(M, 3))
        z = bank.get_state()[:, :3] + rng.normal(scale=30.0, size=(M, 3))
        # ~30% of tracks coast this step (predict only, no measurement).
        mask = rng.random(M) > 0.3

        for j in range(M):
            objs[j].predict(float(dt[j]))
            if mask[j]:
                objs[j].set_measurement_std(tuple(std[j]))
                objs[j].update(z[j])
        bank.predict(dt)
        bank.set_measurement_std(std)
        bank.update(z, mask=mask)

        err = _compare(objs, bank)
        assert err < TOL, f"step {step}: drift {err:.2e} exceeds {TOL:.0e}"
