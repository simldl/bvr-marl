"""
Trigger processing for weapons and countermeasures.
Simple threshold-based trigger logic.
"""


class TriggerProcessor:
    """Process trigger actions for firing and deployment."""

    def __init__(self, trigger_threshold: float = 0.5):
        """
        Initialize trigger processor.

        Args:
            trigger_threshold: Action threshold for trigger activation
        """
        self.trigger_threshold = trigger_threshold

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
        return u_val >= self.trigger_threshold
