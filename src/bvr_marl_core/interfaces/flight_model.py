"""FlightModelPlugin protocol."""

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FlightModelPlugin(Protocol):
    model_id: str

    def compute_acceleration(
        self,
        platform_state: Any,
        command: Any,
        dt: float,
    ) -> np.ndarray:
        """Return 3-D acceleration vector in body frame (m/s²)."""
        ...

    def compute_drag(self, platform_state: Any) -> float:
        """Return drag force magnitude (N)."""
        ...
