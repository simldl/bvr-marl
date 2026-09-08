import numpy as np

from bvr_marl_core.radar.tracking.filter.base_filter import BaseKFFilter


def _inv3_symmetric(S: np.ndarray) -> np.ndarray | None:
    """Closed-form inverse of a well-conditioned 3x3 matrix (returns None if singular)."""
    a, b, c = S[0, 0], S[0, 1], S[0, 2]
    d, e, f = S[1, 0], S[1, 1], S[1, 2]
    g, h, i = S[2, 0], S[2, 1], S[2, 2]
    ca = e * i - f * h
    cb = c * h - b * i
    cc = b * f - c * e
    det = a * ca + d * cb + g * cc
    if not np.isfinite(det) or abs(det) < 1e-300:
        return None
    inv_det = 1.0 / det
    # Writing into a preallocated array is ~18% faster than np.array() over nested
    # Python lists (same float results); this runs once per filter update.
    out = np.empty((3, 3), dtype=float)
    out[0, 0] = ca * inv_det
    out[0, 1] = cb * inv_det
    out[0, 2] = cc * inv_det
    out[1, 0] = (f * g - d * i) * inv_det
    out[1, 1] = (a * i - c * g) * inv_det
    out[1, 2] = (c * d - a * f) * inv_det
    out[2, 0] = (d * h - e * g) * inv_det
    out[2, 1] = (b * g - a * h) * inv_det
    out[2, 2] = (a * e - b * d) * inv_det
    return out


class ConstantVelocityKFFilter(BaseKFFilter):
    """
    Constant-Velocity Kalman Filter.
    State: [x, y, z, vx, vy, vz] in the filter's ENU coordinate system.
    """

    def __init__(
        self,
        initial_measurement: np.ndarray,  # (3,)
        dt: float,
        process_noise_std: float = 25.0,
        measurement_noise_var: float = 1.0,
        R_diag: tuple[float, float, float] | None = None,
    ):
        """
        Constant-Velocity Kalman Filter.
        State: [x, y, z, vx, vy, vz] in the filter's ENU coordinate system.

        ``process_noise_std`` is retained as the public compatibility name, but
        represents the continuous white-acceleration spectral-density amplitude
        in m/s^(3/2), not a discrete acceleration standard deviation in m/s^2.
        """
        self.dt = float(max(1e-6, dt))
        self.x = np.array(
            [
                float(initial_measurement[0]),
                float(initial_measurement[1]),
                float(initial_measurement[2]),
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

        # Measurement covariance
        if R_diag is None:
            self.R = np.diag([measurement_noise_var] * 3)
        else:
            self.R = np.diag(R_diag)

        # Start with a *realistic* initial covariance:
        # - Position seeded from measurement covariance (with a floor)
        # - Velocity very uncertain initially
        pos_var = np.maximum(np.diag(self.R), np.array([25.0, 25.0, 25.0]))  # ≥ 5 m std
        vel_var = np.array([90000.0, 90000.0, 90000.0])  # (300 m/s)^2
        self.P = np.diag(np.concatenate([pos_var, vel_var]))

        # System matrices
        self.F = np.eye(6)
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )

        # Option A: continuous white-acceleration spectral-density amplitude.
        self.process_noise_spectral_amplitude = float(process_noise_std)
        self.sigma_a = self.process_noise_spectral_amplitude  # compatibility alias
        self.Q = np.zeros((6, 6), dtype=float)
        self._applied_dt = None
        self._apply_dt()  # builds F and Q from current dt

    def _apply_dt(self):
        """Update F and Q for the current time step (white-noise acceleration)."""
        dt = self.dt
        # F and Q only depend on dt (spectral amplitude is fixed).
        if dt == self._applied_dt:
            return
        self._applied_dt = dt

        # Only the three coupling entries of F ever change; the rest stays identity.
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        dt2 = dt * dt
        dt3 = dt2 * dt
        # q has units m^2/s^3. The continuous-time integral below produces
        # position/velocity covariance in consistent SI units.
        q = self.process_noise_spectral_amplitude**2

        # Q for each axis (pos, vel): q * [[dt^3/3, dt^2/2],[dt^2/2, dt]]
        q_pp = q * (dt3 / 3.0)
        q_pv = q * (dt2 / 2.0)
        q_vv = q * dt

        Q = self.Q
        # X axis block
        Q[0, 0] = q_pp
        Q[0, 3] = q_pv
        Q[3, 0] = q_pv
        Q[3, 3] = q_vv
        # Y axis block
        Q[1, 1] = q_pp
        Q[1, 4] = q_pv
        Q[4, 1] = q_pv
        Q[4, 4] = q_vv
        # Z axis block
        Q[2, 2] = q_pp
        Q[2, 5] = q_pv
        Q[5, 2] = q_pv
        Q[5, 5] = q_vv

    def set_measurement_std(self, std_xyz: tuple[float, float, float]):
        """Set measurement standard deviations for each axis."""
        sx, sy, sz = std_xyz
        # A preceding common-frame report may have installed a full correlated
        # covariance. Clear those off-diagonal terms before switching back to
        # the diagonal model; retaining them can make R indefinite when the new
        # diagonal variances are smaller.
        R = self.R
        R.fill(0.0)
        R[0, 0] = sx * sx
        R[1, 1] = sy * sy
        R[2, 2] = sz * sz

    def set_measurement_covariance(self, covariance: np.ndarray):
        """Set a full Cartesian measurement covariance, preserving correlations."""
        value = np.asarray(covariance, dtype=float)
        if value.shape != (3, 3):
            raise ValueError(f"Measurement covariance must be 3x3, received {value.shape}.")
        value = 0.5 * (value + value.T)
        if not np.all(np.isfinite(value)) or np.linalg.eigvalsh(value).min() < -1e-9:
            raise ValueError("Measurement covariance must be finite and positive semidefinite.")
        self.R = value.copy()

    def predict(self, dt: float):
        """Predict step."""
        self.dt = float(max(1e-6, dt))
        self._apply_dt()
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        """Update step (Joseph stabilized form)."""
        z = np.asarray(z, dtype=float).reshape(3)
        # H selects the position block, so H@x, H@P@H.T and P@H.T are plain slices.
        y = z - self.x[:3]  # innovation
        S = self.P[:3, :3] + self.R  # innovation covariance
        S[0, 0] += 1e-9  # jitter for numerical safety
        S[1, 1] += 1e-9
        S[2, 2] += 1e-9

        S_inv = _inv3_symmetric(S)
        if S_inv is not None:
            K = self.P[:, :3] @ S_inv  # Kalman gain (reuses S_inv)
            self._last_nis = float(y @ S_inv @ y)  # NIS (reuses S_inv)
        else:
            K = np.zeros((6, 3))
            self._last_nis = 1e6  # penalise, not silently "perfect"

        # State update
        self.x = self.x + (K @ y)

        # Joseph form to keep P symmetric PSD, expanded with H as the position
        # selector:  (I-KH) P (I-KH)^T = (P - K@P[:3,:]) with the same correction
        # applied from the right.
        A = self.P - K @ self.P[:3, :]  # (I - K@H) @ P
        A -= A[:, :3] @ K.T  # ... @ (I - K@H).T
        self.P = A + K @ self.R @ K.T

    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Get current covariance estimate."""
        return self.P.copy()

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.x[3:6].copy()

    def set_state(self, x: np.ndarray, P: np.ndarray | None = None):
        """Set state and optionally covariance."""
        self.x = np.array(x, dtype=float).copy()
        if P is not None:
            self.P = np.array(P, dtype=float).copy()
