"""
Public baseline reward calculator for BVR air combat.

This is a minimal reward baseline provided for public use.
It rewards kills, penalises destruction, and penalises boundary violations.
Users are encouraged to replace this with custom reward functions.
"""

from __future__ import annotations


class RewardCalculator:
    """
    Minimal public reward calculator.

    Provides simple terminal rewards (kill / death / boundary violation)
    as a starting baseline for training. No per-step shaping is applied.
    """

    def __init__(self, **kwargs):
        # Only the three terminal magnitudes are used; all other kwargs are silently ignored
        # so that configs written for extended reward systems still load without error.
        self.kill_reward = float(kwargs.get("kill_reward", 1.0))
        self.destruction_penalty = float(kwargs.get("destruction_penalty", -1.0))
        self.boundary_violation_penalty = float(kwargs.get("boundary_violation_penalty", -1.0))
        self.last_team_reward = float(kwargs.get("last_team_reward", 0.5))

    # ------------------------------------------------------------------
    # Public interface — must match what StepProcessor expects:
    #   returns (total_reward: float, sqi: float, tactical_potential: float, energy_advantage: float)
    # ------------------------------------------------------------------

    def compute_total_reward(self, unit, map_limits, **kwargs) -> tuple[float, float, float, float]:
        """
        Compute a scalar reward for one agent step.

        Returns a 4-tuple (reward, sqi, tactical_potential, energy_advantage).
        The last three values are tracking scalars; this baseline always returns 0.0 for them.
        """
        destroyed = kwargs.get("destroyed", False)
        enemy_kill_count = int(kwargs.get("enemy_kill_count", 0))
        is_last_team = kwargs.get("is_last_team", False)
        died_from_boundary_violation = kwargs.get("died_from_boundary_violation", False)

        R = 0.0

        if enemy_kill_count > 0:
            R += self.kill_reward * enemy_kill_count

        if destroyed:
            if died_from_boundary_violation:
                R += self.boundary_violation_penalty
            else:
                R += self.destruction_penalty

        if is_last_team:
            R += self.last_team_reward

        return float(R), 0.0, 0.0, 0.0

    def get_reward_breakdown(self, unit, map_limits, **kwargs) -> dict[str, float]:
        """Return a flat dict of named reward components (for logging)."""
        reward, _, _, _ = self.compute_total_reward(unit, map_limits, **kwargs)
        return {"total": reward}
