"""WeaponPlugin protocol."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WeaponPlugin(Protocol):
    weapon_type: str
    is_available: bool

    def can_engage(self, shooter_state: Any, target_state: Any) -> bool: ...

    def launch(self, shooter_state: Any, target_state: Any, simulator: Any) -> Any:
        """Create and register a weapon unit; return it."""
        ...

    def reset(self) -> None: ...
