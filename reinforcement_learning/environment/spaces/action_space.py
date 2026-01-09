"""
Backward compatibility layer for action_space module.
Imports from the new modular structure.
"""
import logging

# Import from new modular structure
from .action_space import (
    ActionSpaceManager,
    ActionProcessor,
    EnergyLiftVectorActionProcessor,
)

# Set up module logger
logger = logging.getLogger(__name__)

# Export all classes for backward compatibility
__all__ = [
    'logger',
    'ActionSpaceManager',
    'ActionProcessor',
    'EnergyLiftVectorActionProcessor',
]
