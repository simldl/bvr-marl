"""Action processors for different control modes."""

from bvr_marl_core.rl.environment.spaces.action_space.processors.energy_processor import (
    EnergyProcessor,
)
from bvr_marl_core.rl.environment.spaces.action_space.processors.lift_vector_processor import (
    LiftVectorProcessor,
)
from bvr_marl_core.rl.environment.spaces.action_space.processors.trigger_processor import (
    TriggerProcessor,
)

__all__ = ["EnergyProcessor", "LiftVectorProcessor", "TriggerProcessor"]
