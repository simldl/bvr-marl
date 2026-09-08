"""Vectorized constant-velocity Kalman filter *bank*.

This processes ``M`` independent CV-KF tracks with a handful of batched numpy
calls instead of ``M`` per-object calls, collapsing the per-call dispatch/alloc
overhead that dominates the tiny 6-state filter math.

It is a **drop-in numerical twin** of :class:`ConstantVelocityKFFilter`: every
step performs the same algebra (same white-noise-acceleration ``F``/``Q``, same
closed-form 3x3 innovation inverse with the identical singular-matrix fallback,
the same expanded Joseph covariance update, the same NIS). The differential
harness in ``tests/radar/tracking/test_filter_bank_equivalence.py`` pins the two
together to ``< 1e-9`` relative error.

NOTE: this module is intentionally **not** wired into the tracker / sim yet.
It is pure addition; nothing in
the simulator calls it, so it cannot change any existing behavior.
"""

import numpy as np

# Mirrors ConstantVelocityKFFilter defaults so a bank seeded the same way matches.
_SINGULAR_NIS = 1e6
_S_JITTER = 1e-9


def _inv3_symmetric_batch(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched closed-form inverse of stacked 3x3 matrices.

    Returns ``(S_inv, valid)`` where ``S_inv`` has shape ``(M, 3, 3)`` and
    ``valid`` is a ``(M,)`` bool mask. This replicates the scalar
    ``_inv3_symmetric`` cofactor expansion element-for-element (so the values
    match to floating-point round-off), and flags rows whose determinant is
    non-finite or below ``1e-300`` exactly as the scalar version returns ``None``.

    Invalid rows are given a finite placeholder inverse (computed as if
    ``det == 1``) so no NaN/Inf propagates; callers must zero them out via the
    ``valid`` mask.
    """
    a = S[:, 0, 0]
    b = S[:, 0, 1]
    c = S[:, 0, 2]
    d = S[:, 1, 0]
    e = S[:, 1, 1]
    f = S[:, 1, 2]
    g = S[:, 2, 0]
    h = S[:, 2, 1]
    i = S[:, 2, 2]

    ca = e * i - f * h
    cb = c * h - b * i
    cc = b * f - c * e
    det = a * ca + d * cb + g * cc

    valid = np.isfinite(det) & (np.abs(det) >= 1e-300)
    # Keep the division finite for invalid rows; they are masked off afterwards.
    det_safe = np.where(valid, det, 1.0)
    inv_det = np.where(valid, 1.0 / det_safe, 0.0)

    out = np.empty((S.shape[0], 3, 3), dtype=float)
    out[:, 0, 0] = ca * inv_det
    out[:, 0, 1] = cb * inv_det
    out[:, 0, 2] = cc * inv_det
    out[:, 1, 0] = (f * g - d * i) * inv_det
    out[:, 1, 1] = (a * i - c * g) * inv_det
    out[:, 1, 2] = (c * d - a * f) * inv_det
    out[:, 2, 0] = (d * h - e * g) * inv_det
    out[:, 2, 1] = (b * g - a * h) * inv_det
    out[:, 2, 2] = (a * e - b * d) * inv_det
    return out, valid


class CVFilterBank:
    """A bank of ``M`` constant-velocity Kalman filters advanced together.

    State layout per track matches :class:`ConstantVelocityKFFilter`:
        x : (M, 6)   -> [x, y, z, vx, vy, vz]
        P : (M, 6, 6)
        R : (M, 3)   measurement-noise *variances* on the diagonal (per axis)
        sigma_a : (M,)   continuous white-acceleration spectral-density
            amplitude in m/s^(3/2) (compatibility parameter name)
    """

    def __init__(
        self,
        x: np.ndarray,
        P: np.ndarray,
        sigma_a: np.ndarray,
        R_var: np.ndarray,
    ):
        self.x = np.array(x, dtype=float).reshape(-1, 6)
        M = self.x.shape[0]
        self.P = np.array(P, dtype=float).reshape(M, 6, 6)
        self.sigma_a = np.array(sigma_a, dtype=float).reshape(M)
        self.R_var = np.array(R_var, dtype=float).reshape(M, 3)
        self.last_nis = np.zeros(M, dtype=float)

    @property
    def size(self) -> int:
        return self.x.shape[0]

    def set_measurement_std(self, std_xyz: np.ndarray):
        """Set per-track measurement std on each axis (stored squared as variance)."""
        std = np.asarray(std_xyz, dtype=float).reshape(self.size, 3)
        self.R_var = std * std

    def _build_F_Q(self, dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Continuous white-acceleration F/Q for per-track dt (mirrors _apply_dt)."""
        M = self.size
        F = np.broadcast_to(np.eye(6), (M, 6, 6)).copy()
        F[:, 0, 3] = dt
        F[:, 1, 4] = dt
        F[:, 2, 5] = dt

        dt2 = dt * dt
        dt3 = dt2 * dt
        q = self.sigma_a**2
        q_pp = q * (dt3 / 3.0)
        q_pv = q * (dt2 / 2.0)
        q_vv = q * dt

        Q = np.zeros((M, 6, 6), dtype=float)
        for p, v in ((0, 3), (1, 4), (2, 5)):
            Q[:, p, p] = q_pp
            Q[:, p, v] = q_pv
            Q[:, v, p] = q_pv
            Q[:, v, v] = q_vv
        return F, Q

    def predict(self, dt):
        """Batched predict. ``dt`` may be a scalar or a ``(M,)`` array."""
        dt = np.maximum(1e-6, np.broadcast_to(np.asarray(dt, dtype=float), (self.size,)))
        F, Q = self._build_F_Q(dt)
        self.x = (F @ self.x[:, :, None])[:, :, 0]
        self.P = F @ self.P @ F.transpose(0, 2, 1) + Q

    def update(self, z: np.ndarray, mask: np.ndarray | None = None):
        """Batched measurement update.

        ``z`` is ``(M, 3)``. ``mask`` (``(M,)`` bool) selects which tracks have a
        measurement this step; unmasked rows are left untouched (coast tracks).
        """
        M = self.size
        z = np.asarray(z, dtype=float).reshape(M, 3)

        y = z - self.x[:, :3]  # innovation (M, 3)

        S = self.P[:, :3, :3].copy()
        # S = P[:3,:3] + diag(R) with the same +1e-9 diagonal jitter as the scalar filter.
        idx = np.arange(3)
        S[:, idx, idx] += self.R_var + _S_JITTER

        S_inv, valid = _inv3_symmetric_batch(S)

        K = self.P[:, :, :3] @ S_inv  # (M, 6, 3)
        nis = (y[:, None, :] @ S_inv @ y[:, :, None])[:, 0, 0]

        # Singular innovation covariance -> K=0, penalised NIS (matches scalar path).
        K[~valid] = 0.0
        nis = np.where(valid, nis, _SINGULAR_NIS)

        new_x = self.x + (K @ y[:, :, None])[:, :, 0]

        # Joseph form, expanded with H as the position selector (same as scalar):
        #   A = (I - K H) P ;  P' = A (I - K H)^T + K R K^T
        A = self.P - K @ self.P[:, :3, :]
        A = A - A[:, :, :3] @ K.transpose(0, 2, 1)
        new_P = A + (K * self.R_var[:, None, :]) @ K.transpose(0, 2, 1)

        if mask is None:
            self.x = new_x
            self.P = new_P
            self.last_nis = nis
        else:
            m = np.asarray(mask, dtype=bool).reshape(M)
            self.x[m] = new_x[m]
            self.P[m] = new_P[m]
            self.last_nis[m] = nis[m]

    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()
