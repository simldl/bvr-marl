"""
Action space manager for multi-agent scenarios.
Builds Gym Box spaces for actions.
"""

import numpy as np
from gymnasium.spaces import Box


class ActionSpaceManager:
    """
    Build per-agent Gym Box-Space for actions.
    10 entries: [Ps, n, phi, target, missile_fire, gun_fire, 4xCM]
    Centered flight controls use 0.5 as neutral:
    Ps=0.5 -> hold energy, n=0.5 -> 1g, phi=0.5 -> wings level.
    """

    def __init__(
        self,
        agent_ids: list[str],
        lower: float = 0.0,
        upper: float = 1.0,
        shape: int = 10,
    ):
        """
        Initialize action space manager.

        Args:
            agent_ids: list of agent identifiers
            lower: Lower bound for action values
            upper: Upper bound for action values
            shape: Action dimension (default 10)
        """
        self.agent_ids = agent_ids
        self.lower = lower
        self.upper = upper
        self.shape = shape

        # Template Box
        self._template = Box(self.lower, self.upper, shape=(self.shape,), dtype=np.float32)

        # Per-agent identical Box space
        self.spaces: dict[str, Box] = {aid: self._template for aid in self.agent_ids}

    def get(self, agent_id: str) -> Box:
        """Get action space for a specific agent."""
        return self.spaces[agent_id]

    def all(self) -> dict[str, Box]:
        """Get all action spaces."""
        return self.spaces
