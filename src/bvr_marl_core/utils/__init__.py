"""Public utilities re-exported from submodules for convenient top-level access."""

from bvr_marl_core.utils.geometry import (
    enu_delta_meters,
    rel_position,
    rel_state,
    rel_velocity,
    velocity_components,
)
from bvr_marl_core.utils.simple_config import (
    SimpleConfig,
    apply_overrides,
    load_config,
    save_config,
)


def resolve_relative_path(path_str) -> "str | None":
    """Resolve a user-supplied path string to an absolute path (None stays None)."""
    if path_str is None:
        return None
    from pathlib import Path

    p = Path(str(path_str).strip())
    if p.is_absolute():
        return str(p)
    return str(Path(path_str).resolve())


__all__ = [
    "enu_delta_meters",
    "rel_position",
    "rel_state",
    "rel_velocity",
    "velocity_components",
    "SimpleConfig",
    "apply_overrides",
    "load_config",
    "save_config",
    "resolve_relative_path",
]
