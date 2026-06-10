"""
Minimal reward calculator for BVR air combat.

This package ships the terminal-only baseline ``RewardCalculator``.
For the full modular reward suite (tactical, energy, control, defensive)
use an extension package's ``rl.environment.rewards``.
"""

from .calculator import RewardCalculator

__all__ = ["RewardCalculator"]
