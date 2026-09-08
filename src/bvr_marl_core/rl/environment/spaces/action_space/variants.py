"""
Action processor variants.
Alternative implementations with different configurations.
"""

from bvr_marl_core.rl.environment.spaces.action_space.action_processor import ActionProcessor


class EnergyLiftVectorActionProcessor(ActionProcessor):
    """Alternative energy + lift-vector action space implementation."""

    def __init__(self, simulator, information_mode=None):
        """Initialize with energy space enabled."""
        # ActionProcessor now only supports energy-based control by default
        super().__init__(simulator, information_mode=information_mode)
        # Energy space with simple threshold triggers (no state persistence)
