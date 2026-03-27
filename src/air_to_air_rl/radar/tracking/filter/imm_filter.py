import math

import numpy as np
from numpy.linalg import det

from air_to_air_rl.radar.tracking.filter.base_filter import BaseKFFilter


class IMMFilter(BaseKFFilter):
    """
    Interacting Multiple Model (IMM) filter.
    Combines multiple filters with mode probabilities and transitions.
    """

    def __init__(
        self,
        filters: list[BaseKFFilter],
        transition_matrix: np.ndarray,
        mode_probabilities: list[float],
        measurement_cov: np.ndarray,
    ):
        self.filters = filters
        self.PI = transition_matrix.astype(float).copy()
        self.mu = np.array(mode_probabilities, dtype=float)
        self.mu = self.mu / max(1e-12, np.sum(self.mu))
        self.R = measurement_cov.astype(float).copy()

    def _mix_states(self):
        """Mixing step: compute mixed initial conditions for each filter."""
        M = len(self.filters)
        c = self.PI.T @ self.mu
        c = np.where(c <= 0.0, 1e-9, c)
        mixing = (self.PI * self.mu[:, None]) / c[None, :]

        for j in range(M):
            # Compute mixed mean (always in 6D format)
            xj = np.zeros(6)
            Pj = np.zeros((6, 6))

            for i in range(M):
                xi = self.filters[i].get_state()  # Always returns 6D
                Pi = self.filters[i].get_covariance()  # Always returns 6x6
                xj += mixing[i, j] * xi

            # Compute mixed covariance
            for i in range(M):
                xi = self.filters[i].get_state()
                Pi = self.filters[i].get_covariance()
                d = (xi - xj).reshape(-1, 1)
                Pj += mixing[i, j] * (Pi + d @ d.T)

            # Set state using 6D format (filters handle internal conversion)
            self.filters[j].set_state(xj, Pj)

    def predict(self, dt: float):
        """Predict step for all filters."""
        self._mix_states()
        for f in self.filters:
            f.predict(dt)

    def _likelihood(self, f: BaseKFFilter, z: np.ndarray) -> float:
        """Compute likelihood of measurement z given filter f."""
        x = f.get_state()
        y = z.reshape(3) - x[:3]  # Innovation (position only)
        S = f.get_covariance()[:3, :3] + self.R  # Innovation covariance

        try:
            S_inv = np.linalg.inv(S)
            expo = -0.5 * (y.T @ S_inv @ y)
            denom = math.sqrt(((2 * math.pi) ** 3) * max(1e-12, det(S)))
            return float(math.exp(expo) / max(1e-12, denom))
        except np.linalg.LinAlgError:
            return 1e-12

    def update(self, z: np.ndarray):
        """Update step with mode probability updates."""
        # Compute likelihoods
        lik = np.array([self._likelihood(f, z) for f in self.filters], dtype=float)

        # Update mode probabilities
        c = self.PI.T @ self.mu
        post = lik * c
        s = float(np.sum(post))

        if s > 0:
            self.mu = post / s
        # else keep previous probabilities

        # Update all filters
        for f in self.filters:
            f.update(z)

    def get_state(self) -> np.ndarray:
        """Get IMM state estimate (weighted combination)."""
        out = np.zeros(6)
        for j, f in enumerate(self.filters):
            out += self.mu[j] * f.get_state()
        return out

    def get_covariance(self) -> np.ndarray:
        """Get IMM covariance estimate."""
        out = np.zeros((6, 6))
        xbar = self.get_state()

        for j, f in enumerate(self.filters):
            xj = f.get_state()
            Pj = f.get_covariance()
            d = (xj - xbar).reshape(-1, 1)
            out += self.mu[j] * (Pj + d @ d.T)
        return out

    def get_velocity(self) -> np.ndarray:
        """Get IMM velocity estimate."""
        v = np.zeros(3)
        for j, f in enumerate(self.filters):
            v += self.mu[j] * f.get_velocity()
        return v

    def set_state(self, x: np.ndarray, P: np.ndarray | None = None):
        """Set state for all filters."""
        for f in self.filters:
            f.set_state(x, P)

    def get_last_update_stats(self) -> dict:
        """Return update stats from the dominant (highest-probability) mode."""
        dominant = int(np.argmax(self.mu))
        return self.filters[dominant].get_last_update_stats()

    @property
    def mode_probabilities(self) -> np.ndarray:
        """Get current mode probabilities."""
        return self.mu.copy()
