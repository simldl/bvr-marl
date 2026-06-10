"""Controller protocols for bvr_marl_core.

A Controller is any object that, given simulation state, returns a numpy
action array for a single aircraft agent.  Two flavours exist:

* ``Controller``        — receives an explicit observation vector (neural policy style).
* ``ScriptedController``— reads the world directly through the aircraft/simulator
                           reference it was constructed with (BT / rule-based style).

Both are ``runtime_checkable`` so ``isinstance(obj, Controller)`` works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    pass  # avoid circular imports from simulator types


@runtime_checkable
class Controller(Protocol):
    """Protocol for neural / observation-driven controllers.

    The expected call signature matches ``NeuralWrapper.get_action``.
    """

    def get_action(
        self,
        observation: np.ndarray,
        simulator,
        dt: float,
    ) -> np.ndarray:
        """Return an action array for the current time step.

        Parameters
        ----------
        observation:
            Flat observation vector as produced by the environment's
            observation builder.
        simulator:
            Live ``Simulator`` instance (for reading world state).
        dt:
            Simulation time step in seconds.

        Returns
        -------
        np.ndarray
            Action array whose shape and semantics match the environment's
            action space.
        """
        ...

    def reset(self) -> None:
        """Reset any per-episode state (e.g. hidden LSTM state)."""
        ...


@runtime_checkable
class ScriptedController(Protocol):
    """Protocol for scripted / rule-based controllers.

    The expected call signature matches ``FullTacticalController.get_action``.
    Scripted controllers hold a reference to the aircraft they control and
    read world state directly rather than through an observation vector.
    """

    def get_action(
        self,
        simulator,
        dt: float,
    ) -> np.ndarray:
        """Return an action array for the current time step.

        Parameters
        ----------
        simulator:
            Live ``Simulator`` instance.
        dt:
            Simulation time step in seconds.

        Returns
        -------
        np.ndarray
            Action array.
        """
        ...

    def reset(self) -> None:
        """Reset any per-episode state."""
        ...
