"""
Trigger processing for weapons and countermeasures.
Simple threshold-based trigger logic.
"""


class TriggerProcessor:
    """Process trigger actions for firing and deployment."""

    def __init__(
        self,
        trigger_threshold: float = 0.5,
        per_index_thresholds: dict[int, float] | None = None,
    ):
        """
        Initialize trigger processor.

        Args:
            trigger_threshold: Action threshold for trigger activation
            per_index_thresholds: Optional action-index-specific thresholds
        """
        self.trigger_threshold = float(trigger_threshold)
        self.per_index_thresholds = {
            int(index): float(threshold)
            for index, threshold in (per_index_thresholds or {}).items()
        }

    def set_threshold(self, index: int, threshold: float) -> None:
        """Set a trigger threshold for a specific action index."""
        self.per_index_thresholds[int(index)] = float(threshold)

    def get_threshold(self, index: int) -> float:
        """Return the threshold for an action index."""
        return self.per_index_thresholds.get(int(index), self.trigger_threshold)

    def apply_trigger(self, u_val: float, index: int, state: dict) -> bool:
        """
        Simple threshold trigger: returns True if action >= threshold.

        Args:
            u_val: Action value [0,1]
            index: Action index (for future extensibility)
            state: Agent state dict (unused, for API consistency)

        Returns:
            True if trigger should activate
        """
        return u_val >= self.get_threshold(index)
