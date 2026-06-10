"""ProtocolAdapter protocol — external protocol/interop adapters (DIS, HLA, Link-16, gRPC…)."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProtocolAdapter(Protocol):
    protocol_id: str

    def encode(self, canonical_state: Any) -> Any:
        """Encode canonical domain state to the external protocol format."""
        ...

    def decode(self, message: Any) -> Any:
        """Decode an external message to a canonical command or event."""
        ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...
