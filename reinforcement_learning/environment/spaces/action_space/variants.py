"""
Action processor variants.
Alternative implementations with different configurations.
"""
from .action_processor import ActionProcessor


class EnergyLiftVectorActionProcessor(ActionProcessor):
    """Alternative energy + lift-vector action space implementation."""

    def __init__(self, simulator):
        """Initialize with energy space enabled."""
        super().__init__(simulator, use_speed_control=False, use_energy_space=True)
        # Energy space with simple threshold triggers (no state persistence)
