"""
Modular action space implementation for air combat RL.

This package provides a clean separation of concerns:
- Physics calculations (envelope, drag, energy)
- Automation (missile auto-fire, cooldowns)
- Command processors (energy, lift vector, triggers)
- Utilities (deadzone, target sorting, debug info)
"""

from .action_processor import ActionProcessor
from .manager import ActionSpaceManager
from .simplified_processor import SIMPLIFIED_ACTION_DIM, SimplifiedActionProcessor
from .variants import EnergyLiftVectorActionProcessor

__all__ = [
    "ActionSpaceManager",
    "ActionProcessor",
    "EnergyLiftVectorActionProcessor",
    "SimplifiedActionProcessor",
    "SIMPLIFIED_ACTION_DIM",
]
