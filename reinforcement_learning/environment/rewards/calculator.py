"""
Main reward calculator for BVR air combat.

This module uses terminal/sparse rewards only:
- Kills
- Deaths
- Boundary violations
"""

from typing import Any
from .terminal_rewards import TerminalRewards


class RewardCalculator:
    """
    Simplified reward calculator using only terminal/sparse rewards.
    """

    def __init__(self, **kwargs):
        """
        Initialize terminal reward module.

        Args:
            **kwargs: Terminal reward configuration parameters
        """
        # Initialize reward component module
        self.terminal = TerminalRewards(
            destruction_penalty=kwargs.pop('destruction_penalty', -200.0),
            kill_reward=kwargs.pop('kill_reward', 200.0),
            last_team_reward=kwargs.pop('last_team_reward', 40.0),
            boundary_violation_penalty=kwargs.pop('boundary_violation_penalty', -200.0),
        )

    def compute_total_reward(self, unit, map_limits, **kwargs) -> tuple:
        """
        Compute total reward using only terminal rewards.

        Returns:
            Tuple of (total_reward, 0.0, 0.0, 0.0)
        """
        # Extract parameters
        enemy_kill_count = kwargs.get('enemy_kill_count', 0)
        destroyed = kwargs.get('destroyed', False)
        is_last_team = kwargs.get('is_last_team', False)
        died_from_boundary_violation = kwargs.get('died_from_boundary_violation', False)

        # Handle destroyed unit case
        if unit is None or not hasattr(unit, "position") or unit.position is None:
            R = 0.0
            R += self.terminal.compute_enemy_kill_reward(enemy_kill_count)
            if not died_from_boundary_violation:
                R += self.terminal.compute_destruction_penalty(destroyed)
            else:
                R += self.terminal.boundary_violation_penalty
            return float(R), 0.0, 0.0, 0.0

        # TERMINAL REWARDS
        R_total = 0.0
        R_total += self.terminal.compute_destruction_penalty(destroyed)
        R_total += self.terminal.compute_enemy_kill_reward(enemy_kill_count)
        R_total += self.terminal.compute_team_survival_reward(is_last_team)
        R_total += self.terminal.compute_boundary_violation_penalty(unit)

        return float(R_total), 0.0, 0.0, 0.0

    def get_reward_breakdown(self, unit, map_limits, **kwargs) -> dict:
        """Get detailed breakdown of terminal reward components for analysis/debugging."""
        if unit is None or not hasattr(unit, "position") or unit.position is None:
            return {}

        breakdown = {}

        # Extract parameters
        destroyed = kwargs.get('destroyed', False)
        enemy_kill_count = kwargs.get('enemy_kill_count', 0)
        is_last_team = kwargs.get('is_last_team', False)

        # Terminal rewards
        breakdown['destruction'] = self.terminal.compute_destruction_penalty(destroyed)
        breakdown['kills'] = self.terminal.compute_enemy_kill_reward(enemy_kill_count)
        breakdown['survival'] = self.terminal.compute_team_survival_reward(is_last_team)
        breakdown['boundary_violation'] = self.terminal.compute_boundary_violation_penalty(unit)

        breakdown['total'] = sum(breakdown.values())

        return breakdown
