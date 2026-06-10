"""RewardFunction protocol for bvr_marl_core."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for reward functions.

    Matches the interface of an extension package's reward calculator
    (e.g. ``rl.environment.rewards.RewardCalculator``).
    """

    def compute_total_reward(
        self,
        unit,
        map_limits,
        **kwargs: Any,
    ) -> tuple[float, dict[str, float]]:
        """Compute the total scalar reward and a per-component breakdown.

        Parameters
        ----------
        unit:
            The aircraft unit for which the reward is computed.
        map_limits:
            ``MapLimits`` instance describing scenario boundaries.
        **kwargs:
            Additional step information (kills, hits, sim_time, etc.) as
            accepted by the concrete implementation.

        Returns
        -------
        total : float
            Aggregated scalar reward for this step.
        breakdown : dict[str, float]
            Component-level rewards keyed by name, for logging.
        """
        ...

    def reset(self) -> None:
        """Reset any accumulated per-episode state."""
        ...
