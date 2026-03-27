"""
Deadzone filtering for action smoothing.
Reduces noise in low-magnitude commands.
"""

import numpy as np


class DeadzoneFilter:
    """Apply soft deadzone to zero-centered commands."""

    def __init__(self, deadzone: dict = None, deadzone_lambda: dict = None):
        """
        Initialize deadzone filter.

        Args:
            deadzone: Dict mapping action indices to deadzone width
            deadzone_lambda: Dict mapping action indices to softness parameter
        """
        self.deadzone = deadzone if deadzone is not None else {1: 0.01, 2: 0.01}
        self.deadzone_lambda = deadzone_lambda if deadzone_lambda is not None else {1: 0.3, 2: 0.3}

    def apply_deadzone(self, u_tilde: float, index: int) -> float:
        """
        Apply soft deadzone to zero-centered command.

        Args:
            u_tilde: Zero-centered command value
            index: Action index

        Returns:
            Filtered command value
        """
        if index not in self.deadzone:
            return u_tilde

        d = self.deadzone[index]
        lam = self.deadzone_lambda[index]

        abs_u = abs(u_tilde)
        if abs_u <= d:
            return 0.0

        numerator = abs_u - d
        return np.sign(u_tilde) * numerator / (1.0 + lam * numerator)
