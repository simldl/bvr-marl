"""VisualizerPlugin protocol — decouples rendering from the simulation kernel."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VisualizerPlugin(Protocol):
    def on_reset(self, simulator: Any) -> None:
        """Called when the simulation resets."""
        ...

    def on_tick(self, simulator: Any, events: list[Any]) -> None:
        """Called after each simulation tick."""
        ...

    def on_done(self, simulator: Any) -> None:
        """Called when the simulation episode ends."""
        ...
