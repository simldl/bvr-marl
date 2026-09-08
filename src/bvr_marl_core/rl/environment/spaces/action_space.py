"""
Backward compatibility layer for action_space module.
Imports from the new modular structure.
"""

import logging

# Import from new modular structure
from bvr_marl_core.rl.environment.spaces.action_space import (
    ActionProcessor,
    ActionSpaceManager,
    EnergyLiftVectorActionProcessor,
)

# Set up module logger
logger = logging.getLogger(__name__)

# Export all classes for backward compatibility
__all__ = [
    "logger",
    "ActionSpaceManager",
    "ActionProcessor",
    "EnergyLiftVectorActionProcessor",
]
