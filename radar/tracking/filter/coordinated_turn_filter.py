import math
import numpy as np

from radar.tracking.filter.base_filter import BaseKFFilter


class CoordinatedTurnKFFilter(BaseKFFilter):
    """
    Coordinated-Turn Extended Kalman Filter (optional for IMM).
    
    Internal state: [x, y, z, V, yaw_deg, pitch_deg, yaw_rate_deg, pitch_rate_deg]
    Public state: [x, y, z, vx, vy, vz]
    """
    
    def __init__(self,
                 initial_state: np.ndarray,           # (8,)
                 initial_covariance: np.ndarray,      # (8,8)
                 process_noise_cov: np.ndarray,       # (8,8)
                 measurement_noise_cov: np.ndarray,   # (3,3)
                 dt: float,
                 eps: float = 1e-4):
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
            dx = (V/yaw_r) * (math.sin(yaw + yaw_r*dt) - math.sin(yaw)) * math.cos(pit)
            dy = (V/yaw_r) * (-math.cos(yaw + yaw_r*dt) + math.cos(yaw)) * math.cos(pit)

        if abs(pit_r) < self.eps:
            dz = V * dt * math.sin(pit)
        else:
            dz = (V/pit_r) * (math.sin(pit + pit_r*dt) - math.sin(pit))

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
        """Predict step using EKF."""
        self.dt = float(max(1e-6, dt))
        self.x = self._state_transition(self.x, self.dt)
        F = self._numerical_jacobian(self._state_transition, self.x, self.dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray):
        """Update step."""
        # Measurement model: observe position only
        H = np.zeros((3, 8))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        
        y = z.reshape(3) - self.x[:3]  # Innovation
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        # Update state and covariance
        self.x = self.x + K @ y
        I = np.eye(8)
        self.P = (I - K @ H) @ self.P

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
        """Get covariance in 6x6 format."""
        pos_cov = self.P[:3, :3]
        vel_cov = np.eye(3) * 100.0  # Approximate velocity covariance
        return np.block([
            [pos_cov, np.zeros((3, 3))],
            [np.zeros((3, 3)), vel_cov]
        ])

    def get_velocity(self) -> np.ndarray:
        """Get velocity estimate."""
        _, _, _, V, yaw_deg, pit_deg, *_ = self.x
        yaw = math.radians(yaw_deg)
        pit = math.radians(pit_deg)
        return np.array([
            V * math.cos(pit) * math.cos(yaw),
            V * math.cos(pit) * math.sin(yaw),
            V * math.sin(pit)
        ], dtype=float)

    def set_state(self, x: np.ndarray, P: np.ndarray | None = None):
        """Set state and optionally covariance."""
        if x.shape == (6,):
            # Convert from [x, y, z, vx, vy, vz] to internal state
            self.x[:3] = x[:3]
            V = float(np.linalg.norm(x[3:6]))
            if V > 1e-6:
                yaw = math.degrees(math.atan2(x[4], x[3]))
                pit = math.degrees(math.asin(np.clip(x[5]/V, -1.0, 1.0)))
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