"""
Fully scripted air-to-air combat controller with advanced tactical AI.

This module implements comprehensive air-to-air combat tactics including:
- BVR (Beyond Visual Range) engagement strategies
- Energy management and altitude tactics
- Aspect angle and geometry considerations
- Advanced maneuvers (crank, f-pole, skate, etc.)
- Behavior tree-driven tactical decision making

The scripted controller uses the semi-automated helper for countermeasures
and target selection while providing full flight control automation.
"""

from air_to_air_rl.automation.scripted_control.behavior_tree.tactical_tree import (
    TacticalBehaviorTree,
)
from air_to_air_rl.automation.scripted_control.full_tactical_controller import (
    FullTacticalController,
)
from air_to_air_rl.automation.scripted_control.tactics.bvr_tactics import BVRTactics
from air_to_air_rl.automation.scripted_control.tactics.energy_management import EnergyManager

__all__ = ["FullTacticalController", "TacticalBehaviorTree", "BVRTactics", "EnergyManager"]
