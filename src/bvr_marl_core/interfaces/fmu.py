"""FMUPlugin protocol — Functional Mock-up Unit integration."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FMUPlugin(Protocol):
    fmu_id: str

    def initialize(self, start_time: float, stop_time: float) -> None: ...

    def step(self, dt: float, inputs: dict[str, Any]) -> dict[str, Any]:
        """Advance FMU by dt; return output variables."""
        ...

    def reset(self) -> None: ...

    def terminate(self) -> None: ...
