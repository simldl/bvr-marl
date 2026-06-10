import math

import numpy as np

from bvr_marl_core.radar.tracking.filter.base_filter import BaseKFFilter


class CoordinatedTurnKFFilter(BaseKFFilter):
    """
    Coordinated-Turn Extended Kalman Filter (optional for IMM).

    Internal state: [x, y, z, V, yaw_deg, pitch_deg, yaw_rate_deg, pitch_rate_deg]
    Public state: [x, y, z, vx, vy, vz]
    """

    def __init__(
        self,
        initial_state: np.ndarray,  # (8,)
        initial_covariance: np.ndarray,  # (8,8)
        process_noise_cov: np.ndarray,  # (8,8)
        measurement_noise_cov: np.ndarray,  # (3,3)
        dt: float,
        eps: float = 1e-4,
    ):
        self.x = initial_state.copy()
        self.P = initial_covariance.copy()
        self.Q = process_noise_cov.copy()
        self.R = measurement_noise_cov.copy()
        self.dt = float(dt)
        self.eps = float(eps)

    def _state_transition(self, x, dt):
        """Nonlinear state transition function."""
        x_pos, y_pos, z_pos, V, yaw_deg, pit_deg, yaw_rate_deg, pit_rate_deg = x
        yaw = math.radians(yaw_deg)
        pit = math.radians(pit_deg)
        yaw_r = math.radians(yaw_rate_deg)
        pit_r = math.radians(pit_rate_deg)

        # Handle small turn rates
        if abs(yaw_r) < self.eps:
            dx = V * dt * math.cos(yaw) * math.cos(pit)
            dy = V * dt * math.sin(yaw) * math.cos(pit)
        else:
            dx = (V / yaw_r) * (math.sin(yaw + yaw_r * dt) - math.sin(yaw)) * math.cos(pit)
            dy = (V / yaw_r) * (-math.cos(yaw + yaw_r * dt) + math.cos(yaw)) * math.cos(pit)

        if abs(pit_r) < self.eps:
            dz = V * dt * math.sin(pit)
        else:
            dz = (V / pit_r) * (math.sin(pit + pit_r * dt) - math.sin(pit))

        out = np.zeros_like(x)
        out[0:3] = [x_pos + dx, y_pos + dy, z_pos + dz]
        out[3] = V
        out[4] = yaw_deg + yaw_rate_deg * dt
        out[5] = pit_deg + pit_rate_deg * dt
        out[6] = yaw_rate_deg
        out[7] = pit_rate_deg
        return out

    def _numerical_jacobian(self, f, x, dt, h=1e-5):
        """Compute numerical Jacobian of function f at x."""
        n = len(x)
        J = np.zeros((n, n))
        fx = f(x, dt)
        for j in range(n):
            xh = x.copy()
            xh[j] += h
            J[:, j] = (f(xh, dt) - fx) / h
        return J

    def predict(self, dt: float):
        """Predict step using EKF.

        Correct EKF order:
          1. Save prior state x_{k-1|k-1}.
          2. Compute Jacobian F at the PRIOR (not the predicted state).
          3. Propagate covariance:  P = F P F^T + Q.
          4. Advance mean:          x = f(x_{k-1|k-1}).
        Evaluating F after mutating self.x would linearise around the wrong point.
        """
        self.dt = float(max(1e-6, dt))
        x_prev = self.x.copy()  # step 1: save prior
        F = self._numerical_jacobian(
            self._state_transition, x_prev, self.dt
        )  # step 2: Jacobian at prior
        self.P = F @ self.P @ F.T + self.Q  # step 3: covariance propagation
        self.x = self._state_transition(x_prev, self.dt)  # step 4: mean prediction

    def update(self, z: np.ndarray):
        """Update step (Joseph stabilized form)."""
        # Measurement model: observe position only
        H = np.zeros((3, 8))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0

        y = z.reshape(3) - self.x[:3]  # Innovation
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        S += 1e-9 * np.eye(3)  # jitter for numerical safety

        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ H.T @ S_inv  # Kalman gain (reuses S_inv)
            self._last_nis = float(y @ S_inv @ y)  # NIS (reuses S_inv)
        except np.linalg.LinAlgError:
            K = np.zeros((8, 3))
            self._last_nis = 1e6  # penalise, not silently "perfect"

        # Update state and covariance (Joseph form for symmetry/PSD)
        self.x = self.x + K @ y
        eye = np.eye(8)
        I_KH = eye - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def get_last_update_stats(self) -> dict:
        """Return NIS from the most recent update step, or {} if no update has occurred."""
        if not hasattr(self, "_last_nis"):
            return {}
        return {"nis": self._last_nis}

    def get_state(self) -> np.ndarray:
        """Get state in [x, y, z, vx, vy, vz] format."""
        x_pos, y_pos, z_pos, V, yaw_deg, pit_deg, *_ = self.x
        yaw = math.radians(yaw_deg)
        pit = math.radians(pit_deg)

        vx = V * math.cos(pit) * math.cos(yaw)
        vy = V * math.cos(pit) * math.sin(yaw)
        vz = V * math.sin(pit)

        return np.array([x_pos, y_pos, z_pos, vx, vy, vz], dtype=float)

    def get_covariance(self) -> np.ndarray:
        """Get 6x6 position+velocity covariance in [x,y,z,vx,vy,vz] format.

        The internal state is [x, y, z, V, yaw_deg, pit_deg, yaw_rate_deg, pit_rate_deg].
        The public velocity is a nonlinear function of (V, yaw_deg, pit_deg):
            vx = V cos(pit) cos(yaw)
            vy = V cos(pit) sin(yaw)
            vz = V sin(pit)

        We propagate P_8x8 through the Jacobian J (6×8) of get_state() w.r.t. self.x:
            P_6x6 = J @ P_8x8 @ J^T

        This preserves cross-covariances between position and velocity and accounts for
        speed/angle coupling — unlike the discarded isotropic approximation.
        """
        d = math.pi / 180.0  # degree-to-radian conversion for angle columns
        _, _, _, V, yaw_deg, pit_deg, *_ = self.x
        yaw = math.radians(yaw_deg)
        pit = math.radians(pit_deg)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        cos_pit = math.cos(pit)
        sin_pit = math.sin(pit)

        # J[i, j] = d(output_i) / d(internal_state_j)
        # output: [x, y, z, vx, vy, vz]    (6 rows)
        # state:  [x, y, z, V, yaw_deg, pit_deg, yaw_rate_deg, pit_rate_deg]  (8 cols)
        J = np.zeros((6, 8), dtype=float)

        # Position rows: direct copy
        J[0, 0] = 1.0  # dx/dx
        J[1, 1] = 1.0  # dy/dy
        J[2, 2] = 1.0  # dz/dz

        # vx = V cos(pit) cos(yaw)
        J[3, 3] = cos_pit * cos_yaw  # dvx/dV
        J[3, 4] = -V * cos_pit * sin_yaw * d  # dvx/d(yaw_deg)
        J[3, 5] = -V * sin_pit * cos_yaw * d  # dvx/d(pit_deg)

        # vy = V cos(pit) sin(yaw)
        J[4, 3] = cos_pit * sin_yaw  # dvy/dV
        J[4, 4] = V * cos_pit * cos_yaw * d  # dvy/d(yaw_deg)
        J[4, 5] = -V * sin_pit * sin_yaw * d  # dvy/d(pit_deg)

        # vz = V sin(pit)
        J[5, 3] = sin_pit  # dvz/dV
        J[5, 4] = 0.0  # dvz/d(yaw_deg)  = 0
        J[5, 5] = V * cos_pit * d  # dvz/d(pit_deg)

        P6 = J @ self.P @ J.T
        # Enforce symmetry (numerical noise from J @ P @ J^T)
        return 0.5 * (P6 + P6.T)

    def get_velocity(self) -> np.ndarray:
        """Get velocity estimate."""
        _, _, _, V, yaw_deg, pit_deg, *_ = self.x
        yaw = math.radians(yaw_deg)
        pit = math.radians(pit_deg)
        return np.array(
            [
                V * math.cos(pit) * math.cos(yaw),
                V * math.cos(pit) * math.sin(yaw),
                V * math.sin(pit),
            ],
            dtype=float,
        )

    def set_state(self, x: np.ndarray, P: np.ndarray | None = None):
        """Set state and optionally covariance."""
        if x.shape == (6,):
            # Convert from [x, y, z, vx, vy, vz] to internal state
            self.x[:3] = x[:3]
            V = float(np.linalg.norm(x[3:6]))
            if V > 1e-6:
                yaw = math.degrees(math.atan2(x[4], x[3]))
                pit = math.degrees(math.asin(np.clip(x[5] / V, -1.0, 1.0)))
            else:
                yaw = pit = 0.0
            self.x[3:6] = [V, yaw, pit]
        else:
            self.x = x.copy()

        if P is not None:
            if P.shape == (6, 6):
                # Convert 6x6 covariance to 8x8 for internal use
                P_new = np.eye(8) * 100.0  # Default for unmapped dimensions
                P_new[:3, :3] = P[:3, :3]  # Position covariance
                # Approximate velocity covariance mapping
                P_new[3, 3] = np.trace(P[3:, 3:]) / 3.0  # Speed variance
                P_new[4:6, 4:6] = np.eye(2) * 50.0  # Angle variances
                P_new[6:8, 6:8] = np.eye(2) * 10.0  # Rate variances
                self.P = P_new
            else:
                self.P = P.copy()
