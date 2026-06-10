"""
Action processor variants.
Alternative implementations with different configurations.
"""

from .action_processor import ActionProcessor


class EnergyLiftVectorActionProcessor(ActionProcessor):
    """Alternative energy + lift-vector action space implementation."""

    def __init__(self, simulator):
        """Initialize with energy space enabled."""
        # ActionProcessor now only supports energy-based control by default
        super().__init__(simulator)
        # Energy space with simple threshold triggers (no state persistence)
