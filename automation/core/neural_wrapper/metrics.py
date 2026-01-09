"""
Performance metrics tracking for neural wrapper.
Monitors action usage and automation performance.
"""
from typing import Dict, Any
import numpy as np


class MetricsTracker:
    """Track performance metrics for wrapper system."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {
            'manual_actions_taken': 0,
            'automated_actions_taken': 0,
            'countermeasures_deployed': 0,
            'targets_selected': 0
        }

    def update_from_action(self, complete_action: np.ndarray):
        """
        Update metrics based on complete action.

        Args:
            complete_action: The full action array
        """
        self.metrics['manual_actions_taken'] += 1

        # Track target selection
        if complete_action[3] > 0.1:
            self.metrics['targets_selected'] += 1
            self.metrics['automated_actions_taken'] += 1

        # Track countermeasures
        if np.any(complete_action[6:10] > 0.5):
            self.metrics['countermeasures_deployed'] += 1
            self.metrics['automated_actions_taken'] += 1

    def get_metrics(self) -> Dict[str, int]:
        """Get current metrics."""
        return self.metrics.copy()

    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            'manual_actions_taken': 0,
            'automated_actions_taken': 0,
            'countermeasures_deployed': 0,
            'targets_selected': 0
        }
