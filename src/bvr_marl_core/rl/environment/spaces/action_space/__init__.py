"""
Modular action space implementation for air combat RL.

This package provides a clean separation of concerns:
- Physics calculations (envelope, drag, energy)
- Automation (missile auto-fire, cooldowns)
- Command processors (energy, lift vector, triggers)
- Utilities (deadzone, target sorting, debug info)
"""

from bvr_marl_core.rl.environment.spaces.action_space.action_processor import ActionProcessor
from bvr_marl_core.rl.environment.spaces.action_space.manager import (
    ACTION_SCHEMA_VERSION,
    BASE_ACTION_DIM,
    EMCON_ACTION_ENABLED,
    FULL_ACTION_DIM,
    ActionSpaceManager,
    emcon_action_dim,
)
from bvr_marl_core.rl.environment.spaces.action_space.simplified_processor import (
    SIMPLIFIED_ACTION_DIM,
    SimplifiedActionProcessor,
)
from bvr_marl_core.rl.environment.spaces.action_space.variants import (
    EnergyLiftVectorActionProcessor,
)

__all__ = [
    "ActionSpaceManager",
    "ActionProcessor",
    "EnergyLiftVectorActionProcessor",
    "SimplifiedActionProcessor",
    "SIMPLIFIED_ACTION_DIM",
    "FULL_ACTION_DIM",
    "BASE_ACTION_DIM",
    "EMCON_ACTION_ENABLED",
    "ACTION_SCHEMA_VERSION",
    "emcon_action_dim",
]
