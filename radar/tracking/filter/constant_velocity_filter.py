import numpy as np
from numpy.linalg import inv
from typing import Tuple

from radar.tracking.filter.base_filter import BaseKFFilter


class ConstantVelocityKFFilter(BaseKFFilter):
    """
    Constant-Velocity Kalman Filter.
    State: [x, y, z, vx, vy, vz] in the filter's ENU coordinate system.
    """
    
    def __init__(
        self,
        initial_measurement: np.ndarray,  # (3,)
        dt: float,
        process_noise_std: float = 25.0,      # interpreted as (sigma_a) in m/s^2
        measurement_noise_var: float = 1.0,
        R_diag: Tuple[float, float, float] | None = None
    ):
        """
        Constant-Velocity Kalman Filter.
        State: [x, y, z, vx, vy, vz] in the filter's ENU coordinate system.
        """
        self.dt = float(max(1e-6, dt))
        self.x = np.array([
            float(initial_measurement[0]),
            float(initial_measurement[1]),
            float(initial_measurement[2]),
            0.0, 0.0, 0.0
        ], dtype=float)

        # Measurement covariance
        if R_diag is None:
            self.R = np.diag([measurement_noise_var]*3)
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
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # Process model: white-noise acceleration with sigma_a
        self.sigma_a = float(process_noise_std)
        self._apply_dt()  # builds F and Q from current dt


    def _apply_dt(self):
        """Update F and Q for the current time step (white-noise acceleration)."""
        dt = self.dt
        self.F[:] = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        dt2 = dt * dt
        dt3 = dt2 * dt
        q = (self.sigma_a ** 2)

        # Q for each axis (pos, vel): q * [[dt^3/3, dt^2/2],[dt^2/2, dt]]
        Q_axis = np.array([[dt3/3.0, dt2/2.0],
                        [dt2/2.0, dt]], dtype=float)

        self.Q = np.zeros((6, 6), dtype=float)
        # X axis block
        self.Q[0, 0] = q * Q_axis[0, 0]
        self.Q[0, 3] = q * Q_axis[0, 1]
        self.Q[3, 0] = q * Q_axis[1, 0]
        self.Q[3, 3] = q * Q_axis[1, 1]
        # Y axis block
        self.Q[1, 1] = q * Q_axis[0, 0]
        self.Q[1, 4] = q * Q_axis[0, 1]
        self.Q[4, 1] = q * Q_axis[1, 0]
        self.Q[4, 4] = q * Q_axis[1, 1]
        # Z axis block
        self.Q[2, 2] = q * Q_axis[0, 0]
        self.Q[2, 5] = q * Q_axis[0, 1]
        self.Q[5, 2] = q * Q_axis[1, 0]
        self.Q[5, 5] = q * Q_axis[1, 1]


    def set_measurement_std(self, std_xyz: Tuple[float, float, float]):
        """Set measurement standard deviations for each axis."""
        sx, sy, sz = std_xyz
        self.R = np.diag([sx*sx, sy*sy, sz*sz])

    def predict(self, dt: float):
        """Predict step."""
        self.dt = float(max(1e-6, dt))
        self._apply_dt()
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        """Update step (Joseph stabilized form)."""
        z = np.asarray(z, dtype=float).reshape(3)
        y = z - (self.H @ self.x)                           # innovation
        S = self.H @ self.P @ self.H.T + self.R             # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)            # Kalman gain

        # State update
        self.x = self.x + (K @ y)

        # Joseph form to keep P symmetric PSD
        I = np.eye(6)
        I_KH = (I - K @ self.H)
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

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