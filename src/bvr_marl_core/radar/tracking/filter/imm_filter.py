import numpy as np

from bvr_marl_core.radar.tracking.filter.base_filter import BaseKFFilter


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
        if not filters:
            raise ValueError("IMM requires at least one component filter.")
        self.filters = filters
        self.PI = np.asarray(transition_matrix, dtype=float).copy()
        model_count = len(filters)
        if self.PI.shape != (model_count, model_count):
            raise ValueError("IMM transition matrix must be square with one row per model.")
        if np.any(self.PI < 0.0) or not np.allclose(self.PI.sum(axis=1), 1.0):
            raise ValueError("IMM transition rows must be nonnegative and sum to one.")
        self.mu = np.array(mode_probabilities, dtype=float)
        if self.mu.shape != (model_count,) or np.any(self.mu < 0.0):
            raise ValueError("IMM mode probabilities must be nonnegative and match the models.")
        self.mu = self.mu / max(1e-12, np.sum(self.mu))
        self.R = measurement_cov.astype(float).copy()

    def _mix_states(self):
        """Mixing step: compute mixed initial conditions for each filter."""
        M = len(self.filters)
        c = self.PI.T @ self.mu
        c = np.where(c <= 0.0, 1e-9, c)
        mixing = (self.PI * self.mu[:, None]) / c[None, :]

        states = np.stack([model.get_state() for model in self.filters])
        covariances = np.stack([model.get_covariance() for model in self.filters])

        for j in range(M):
            weights = mixing[:, j]
            xj = np.einsum("i,ij->j", weights, states)
            deltas = states - xj
            Pj = np.einsum("i,ijk->jk", weights, covariances) + np.einsum(
                "i,ij,ik->jk", weights, deltas, deltas
            )

            # Set state using 6D format (filters handle internal conversion)
            self.filters[j].set_state(xj, Pj)

    def predict(self, dt: float):
        """Predict step for all filters."""
        self._mix_states()
        for f in self.filters:
            f.predict(dt)

    def _log_likelihood(self, f: BaseKFFilter, z: np.ndarray) -> float:
        """Compute a stable log likelihood of measurement ``z`` for one mode."""
        x = f.get_state()
        y = z.reshape(3) - x[:3]  # Innovation (position only)
        S = f.get_covariance()[:3, :3] + self.R  # Innovation covariance

        try:
            sign, log_determinant = np.linalg.slogdet(S)
            if sign <= 0.0:
                return float("-inf")
            nis = float(y @ np.linalg.solve(S, y))
            return -0.5 * (3.0 * np.log(2.0 * np.pi) + log_determinant + nis)
        except np.linalg.LinAlgError:
            return float("-inf")

    def update(self, z: np.ndarray):
        """Update step with mode probability updates."""
        c = self.PI.T @ self.mu
        log_post = np.asarray(
            [self._log_likelihood(f, z) for f in self.filters], dtype=float
        ) + np.log(np.maximum(c, 1e-300))
        maximum = float(np.max(log_post))
        if np.isfinite(maximum):
            posterior = np.exp(log_post - maximum)
            self.mu = posterior / posterior.sum()

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

    def set_measurement_covariance(self, covariance: np.ndarray):
        value = np.asarray(covariance, dtype=float)
        self.R = value.copy()
        for model in self.filters:
            model.set_measurement_covariance(value)

    def set_measurement_std(self, std_xyz: tuple[float, float, float]):
        values = np.asarray(std_xyz, dtype=float)
        self.set_measurement_covariance(np.diag(values * values))

    def get_last_update_stats(self) -> dict:
        """Return update stats from the dominant (highest-probability) mode."""
        dominant = int(np.argmax(self.mu))
        return self.filters[dominant].get_last_update_stats()

    @property
    def mode_probabilities(self) -> np.ndarray:
        """Get current mode probabilities."""
        return self.mu.copy()
