"""
Minimal reward calculator for BVR air combat.

This package ships the terminal-only baseline ``RewardCalculator``.
For the full modular reward suite (tactical, energy, control, defensive)
use an extension package's ``rl.environment.rewards``.
"""

from bvr_marl_core.rl.environment.rewards.calculator import RewardCalculator
from bvr_marl_core.rl.environment.rewards.information import RewardInformationClass

__all__ = ["RewardCalculator", "RewardInformationClass"]
