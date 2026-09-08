"""Cartesian coordinated-turn extended Kalman filter."""

from __future__ import annotations

import numpy as np

from bvr_marl_core.radar.tracking.filter.base_filter import BaseKFFilter
from bvr_marl_core.radar.tracking.filter.constant_velocity_filter import _inv3_symmetric


class CoordinatedTurnKFFilter(BaseKFFilter):
    """Seven-state horizontal coordinated-turn EKF.

    Internal state is ``[x, y, z, vx, vy, vz, omega]`` in SI units, with
    ``omega`` in rad/s. The public state/covariance contract is the leading
    Cartesian six-state block, exactly matching the production CV filter.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise_cov: np.ndarray,
        measurement_noise_cov: np.ndarray,
        dt: float,
        eps: float = 1e-8,
    ):
        self.x = np.asarray(initial_state, dtype=float).reshape(7).copy()
        self.P = np.asarray(initial_covariance, dtype=float).reshape(7, 7).copy()
        self.Q = np.asarray(process_noise_cov, dtype=float).reshape(7, 7).copy()
        self.R = np.asarray(measurement_noise_cov, dtype=float).reshape(3, 3).copy()
        self.dt = float(max(1e-6, dt))
        self.eps = float(max(0.0, eps))

    def _turn_terms(self, omega: float, dt: float) -> tuple[float, float, float, float]:
        angle = omega * dt
        if abs(omega) <= self.eps:
            # Smooth Taylor form at omega=0; unlike a straight-line branch, this
            # retains the Jacobian sensitivity needed to estimate a nascent turn.
            angle2 = angle * angle
            a = dt * (1.0 - angle2 / 6.0)
            b = 0.5 * omega * dt * dt * (1.0 - angle2 / 12.0)
            c = 1.0 - 0.5 * angle2
            s = angle * (1.0 - angle2 / 6.0)
            return a, b, c, s
        return (
            float(np.sin(angle) / omega),
            float((1.0 - np.cos(angle)) / omega),
            float(np.cos(angle)),
            float(np.sin(angle)),
        )

    def _state_transition(self, state: np.ndarray, dt: float) -> np.ndarray:
        x, y, z, vx, vy, vz, omega = state
        a, b, cosine, sine = self._turn_terms(float(omega), float(dt))
        return np.array(
            [
                x + a * vx - b * vy,
                y + b * vx + a * vy,
                z + dt * vz,
                cosine * vx - sine * vy,
                sine * vx + cosine * vy,
                vz,
                omega,
            ],
            dtype=float,
        )

    def _numerical_jacobian(self, state: np.ndarray, dt: float) -> np.ndarray:
        baseline = self._state_transition(state, dt)
        jacobian = np.empty((7, 7), dtype=float)
        for column in range(7):
            step = 1e-5 * max(1.0, abs(float(state[column])))
            perturbed = state.copy()
            perturbed[column] += step
            jacobian[:, column] = (self._state_transition(perturbed, dt) - baseline) / step
        return jacobian

    def predict(self, dt: float):
        self.dt = float(max(1e-6, dt))
        prior = self.x.copy()
        transition = self._numerical_jacobian(prior, self.dt)
        self.x = self._state_transition(prior, self.dt)
        self.P = transition @ self.P @ transition.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z: np.ndarray):
        innovation = np.asarray(z, dtype=float).reshape(3) - self.x[:3]
        innovation_covariance = self.P[:3, :3] + self.R
        innovation_covariance = innovation_covariance.copy()
        innovation_covariance.flat[::4] += 1e-9
        inverse = _inv3_symmetric(innovation_covariance)
        if inverse is None:
            gain = np.zeros((7, 3), dtype=float)
            self._last_nis = 1e6
        else:
            gain = self.P[:, :3] @ inverse
            self._last_nis = float(innovation @ inverse @ innovation)
        self.x += gain @ innovation
        a = self.P - gain @ self.P[:3, :]
        a -= a[:, :3] @ gain.T
        self.P = a + gain @ self.R @ gain.T
        self.P = 0.5 * (self.P + self.P.T)

    def get_state(self) -> np.ndarray:
        return self.x[:6].copy()

    def get_covariance(self) -> np.ndarray:
        return self.P[:6, :6].copy()

    def get_velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    def set_state(self, x: np.ndarray, P: np.ndarray | None = None):
        value = np.asarray(x, dtype=float)
        if value.shape == (6,):
            self.x[:6] = value
        elif value.shape == (7,):
            self.x = value.copy()
        else:
            raise ValueError(f"CT state must be length 6 or 7, received {value.shape}.")
        if P is not None:
            covariance = np.asarray(P, dtype=float)
            if covariance.shape == (6, 6):
                self.P[:6, :6] = covariance
                self.P[6, :6] = 0.0
                self.P[:6, 6] = 0.0
            elif covariance.shape == (7, 7):
                self.P = covariance.copy()
            else:
                raise ValueError(f"CT covariance must be 6x6 or 7x7, received {covariance.shape}.")

    def set_measurement_covariance(self, covariance: np.ndarray):
        value = np.asarray(covariance, dtype=float)
        if value.shape != (3, 3):
            raise ValueError(f"Measurement covariance must be 3x3, received {value.shape}.")
        value = 0.5 * (value + value.T)
        if not np.all(np.isfinite(value)) or np.linalg.eigvalsh(value).min() < -1e-9:
            raise ValueError("Measurement covariance must be finite and positive semidefinite.")
        self.R = value.copy()

    def set_measurement_std(self, std_xyz: tuple[float, float, float]):
        values = np.asarray(std_xyz, dtype=float)
        self.R = np.diag(values * values)
