import numpy as np


class BaseKFFilter:
    """Base interface for Kalman filter implementations."""

    # Tracker bookkeeping: number of consecutive ticks this filter went without a
    # measurement. The TrackerManager assigns it per-instance; the class-level
    # default lets hot-path readers (prune, confidence, coast increment) use direct
    # attribute access instead of getattr(kf, "missed_updates", 0.0) on every track.
    missed_updates: float = 0.0

    def predict(self, dt: float):
        """Predict step of the filter."""
        ...

    def update(self, z: np.ndarray):
        """Update step of the filter."""
        ...

    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        ...

    def get_covariance(self) -> np.ndarray:
        """Get current covariance estimate."""
        ...

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        ...

    def set_state(self, x: np.ndarray, P: np.ndarray | None = None):
        """Set state and optionally covariance."""
        ...

    def set_measurement_covariance(self, covariance: np.ndarray):
        """Set a full Cartesian position-measurement covariance."""
        ...

    def set_measurement_std(self, std_xyz: tuple[float, float, float]):
        """Set diagonal position-measurement standard deviations."""
        ...

    def get_last_update_stats(self) -> dict:
        """Return NIS from the most recent update, if one has occurred."""
        return {"nis": self._last_nis} if hasattr(self, "_last_nis") else {}
