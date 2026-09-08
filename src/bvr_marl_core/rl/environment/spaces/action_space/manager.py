"""
Action space manager for multi-agent scenarios.
Builds Gym Box spaces for actions.
"""

import numpy as np
from gymnasium.spaces import Box

# EMCON radar on/off is implemented end-to-end (silent radar + RWR coupling), but
# for the initial experiments the agents must not control it — the radar is always
# ON. Flip this single flag to True to expose the radar toggle as action index 9;
# nothing else needs editing (env action space and training config both derive from
# FULL_ACTION_DIM below).
EMCON_ACTION_ENABLED = False
# [Ps, n, phi, missile_fire, target, gun_fire, flares, chaff, decoys]. ECM was
# removed from the action space (it was a no-op — nothing consulted ecm_active).
BASE_ACTION_DIM = 9
FULL_ACTION_DIM = BASE_ACTION_DIM + (1 if EMCON_ACTION_ENABLED else 0)


# Action-schema version. Bump when the action-vector layout changes (e.g. enabling
# the EMCON channel), so a checkpoint refuses to load into a mismatched action space.
ACTION_SCHEMA_VERSION = "1"


def emcon_action_dim(enabled: bool | None = None) -> int:
    """Action-vector width with/without the EMCON radar toggle (action index 9).

    Config-driven replacement for reading the module-level ``EMCON_ACTION_ENABLED``
    constant: pass ``env.emcon_action_enabled`` so a run enables the radar action
    without a source edit. ``None`` falls back to the module default.
    """
    if enabled is None:
        enabled = EMCON_ACTION_ENABLED
    return BASE_ACTION_DIM + (1 if enabled else 0)


class ActionSpaceManager:
    """
    Build per-agent Gym Box-Space for actions.
    Base 9 entries: [Ps, n, phi, missile_fire, target, gun_fire, flares, chaff,
    decoys]; with EMCON enabled a 10th entry is the radar on/off toggle (silent
    only when >=0.75, so it stays ON by default).
    Centered flight controls use 0.5 as neutral:
    Ps=0.5 -> hold energy, n=0.5 -> 1g, phi=0.5 -> wings level.
    """

    def __init__(
        self,
        agent_ids: list[str],
        lower: float = 0.0,
        upper: float = 1.0,
        shape: int = FULL_ACTION_DIM,
    ):
        """
        Initialize action space manager.

        Args:
            agent_ids: list of agent identifiers
            lower: Lower bound for action values
            upper: Upper bound for action values
            shape: Action dimension (default 9)
        """
        self.agent_ids = agent_ids
        self.lower = lower
        self.upper = upper
        self.shape = shape

        self._template = Box(self.lower, self.upper, shape=(self.shape,), dtype=np.float32)

        # Per-agent identical Box space
        self.spaces: dict[str, Box] = {aid: self._template for aid in self.agent_ids}

    def get(self, agent_id: str) -> Box:
        """Get action space for a specific agent."""
        return self.spaces[agent_id]

    def all(self) -> dict[str, Box]:
        """Get all action spaces."""
        return self.spaces
