"""
Modular action space implementation for air combat RL.

This package provides a clean separation of concerns:
- Physics calculations (envelope, drag, energy)
- Automation (missile auto-fire, cooldowns)
- Command processors (energy, lift vector, triggers)
- Utilities (deadzone, target sorting, debug info)
"""
from .manager import ActionSpaceManager
from .action_processor import ActionProcessor
from .variants import EnergyLiftVectorActionProcessor

__all__ = [
    'ActionSpaceManager',
    'ActionProcessor',
    'EnergyLiftVectorActionProcessor',
]
