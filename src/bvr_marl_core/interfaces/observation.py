"""ObservationHook protocol for bvr_marl_core."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ObservationHook(Protocol):
    """Optional post-processing hook applied to raw observations.

    Allows an extension package to augment or normalise observations produced
    by the core observation builders without modifying core code.

    The hook is applied as a final pass after the core builders have
    assembled the raw feature vector.
    """

    def process(
        self,
        observation: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        """Transform or augment the raw observation vector.

        Parameters
        ----------
        observation:
            Raw observation array as assembled by the core builders.
        context:
            Arbitrary context dictionary (e.g. ``{"aircraft": ac, "dt": dt}``).

        Returns
        -------
        np.ndarray
            Processed observation array (may be the same object if no
            transformation is needed, or a new array).
        """
        ...

    def reset(self) -> None:
        """Reset any stateful normalisation buffers for a new episode."""
        ...
