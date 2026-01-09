"""
Reward calculator for BVR air combat.

This module uses terminal/sparse rewards only:
- Terminal rewards (kills, deaths, boundary violations, team survival)

The reward structure encourages learning through sparse feedback
from mission outcomes rather than detailed reward shaping.
"""

# Import main reward calculator
from .calculator import RewardCalculator

# Import terminal rewards component
from .terminal_rewards import TerminalRewards

__all__ = [
    'RewardCalculator',
    'TerminalRewards',
]
