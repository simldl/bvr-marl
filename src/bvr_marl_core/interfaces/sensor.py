"""SensorPlugin protocol — any sensor that can be attached to a platform."""

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SensorPlugin(Protocol):
    sensor_type: str

    def update(self, platform_state: Any, simulator: Any, dt: float) -> None:
        """Advance sensor state by dt seconds."""
        ...

    def get_detections(self) -> list[Any]:
        """Return current detection list."""
        ...

    def reset(self) -> None: ...
