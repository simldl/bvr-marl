"""ControllerFactory protocol for bvr_marl_core."""

from __future__ import annotations

from typing import Any, Protocol

from bvr_marl_core.interfaces.controller import Controller


class ControllerFactory(Protocol):
    """Protocol for objects that create Controller instances.

    Decouples the environment/simulator from concrete controller classes so
    that different controllers (neural, scripted, random) can be swapped in
    without modifying core code.
    """

    def create(
        self,
        aircraft,
        simulator=None,
        **kwargs: Any,
    ) -> Controller:
        """Instantiate and return a configured controller for ``aircraft``.

        Parameters
        ----------
        aircraft:
            The aircraft instance this controller will drive.
        simulator:
            Optional live ``Simulator`` reference (required by some controllers).
        **kwargs:
            Additional controller-specific configuration.

        Returns
        -------
        Controller
            A ready-to-use controller instance.
        """
        ...
