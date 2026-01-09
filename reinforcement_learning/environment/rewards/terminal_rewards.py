"""
Terminal reward components for BVR combat.

These are sparse rewards given at episode termination or major events:
- Destruction penalties (being shot down)
- Kill rewards (shooting down enemies)
- Team survival bonuses
- Boundary violations
"""

import numpy as np
from typing import Optional


class TerminalRewards:
    """Handles terminal/sparse reward events."""

    def __init__(
        self,
        destruction_penalty: float = -200.0,
        kill_reward: float = 200.0,
        last_team_reward: float = 40.0,
        boundary_violation_penalty: float = -200.0,
    ):
        self.destruction_penalty_value = float(destruction_penalty)
        self.kill_reward_value = float(kill_reward)
        self.last_team_reward = float(last_team_reward)
        self.boundary_violation_penalty = float(boundary_violation_penalty)

    def compute_destruction_penalty(self, destroyed: bool) -> float:
        """Penalty for being destroyed."""
        return self.destruction_penalty_value if destroyed else 0.0

    def compute_enemy_kill_reward(self, enemy_kill_count: int) -> float:
        """Reward for killing enemies."""
        return self.kill_reward_value * enemy_kill_count

    def compute_team_survival_reward(self, is_last_team: bool) -> float:
        """Bonus for being the last team standing."""
        return self.last_team_reward if is_last_team else 0.0

    def compute_boundary_violation_penalty(self, unit) -> float:
        """Penalty for violating map boundaries."""
        if unit is None or not hasattr(unit, "position"):
            return 0.0

        # Check if unit has violated boundaries
        if hasattr(unit, "boundary_violation_active") and unit.boundary_violation_active:
            return self.boundary_violation_penalty

        return 0.0
