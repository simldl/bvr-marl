"""Configuration loading utilities for bvr-marl-core.

All config search roots are anchored to the bvr-marl-core package layout.
The neutral path helper ``resolve_relative_path`` is re-exported from
``bvr_marl_core.utils`` for callers that expect to find it here.
"""

from pathlib import Path

import yaml

from bvr_marl_core.utils import resolve_relative_path
from bvr_marl_core.utils.paths import (
    core_project_root,
    core_root,
    rl_configs_root,
    shared_configs_root,
    visualization_configs_root,
)
from bvr_marl_core.utils.paths import (
    gui_config as _gui_config_path,
)

__all__ = [
    "resolve_training_config",
    "load_train_config",
    "load_gui_config",
    "load_viz_config",
    "load_shared_config",
    # re-export neutral helper from core for callers that expect it here
    "resolve_relative_path",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_training_path(name_or_path: "str | Path", default_dir: Path) -> Path:
    """Resolve a training config name or path to an absolute Path."""
    p = Path(name_or_path)
    if p.is_absolute():
        return p
    if "/" in str(name_or_path) or "\\" in str(name_or_path):
        return core_project_root() / p
    stem = p.stem if p.suffix else str(p)
    candidates = [
        core_project_root() / "configs" / "training" / f"{stem}.yaml",
        default_dir / f"{stem}.yaml",
        default_dir / f"train_config_{stem}.yaml",  # legacy name fallback
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return default_dir / f"{stem}.yaml"


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


def resolve_training_config(name_or_path: "str | Path") -> Path:
    """Resolve a training config name/path to an absolute Path (file may not exist)."""
    return _resolve_training_path(name_or_path, rl_configs_root())


def load_train_config(config_path=None) -> dict:
    """Load training configuration.

    Args:
        config_path: Absolute path, relative path, or bare config name.
                     Defaults to ``configs/training/basic.yaml`` at the project
                     root, falling back to legacy package ``rl/configs/`` only
                     when present.
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        default = rl_configs_root() / "train_config.yaml"
        top_level = core_project_root() / "configs" / "training" / "basic.yaml"
        full_path = top_level if top_level.exists() else default
    else:
        full_path = _resolve_training_path(config_path, rl_configs_root())

    with open(full_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# GUI config
# ---------------------------------------------------------------------------


def load_gui_config() -> dict:
    """Load GUI default configuration.

    Searches ``configs/gui/default.yaml`` at the project root first;
    falls back to the package default.  Returns an empty dict if neither exists.
    """
    full_path = _gui_config_path()
    if not full_path.exists():
        return {}
    with open(full_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config else {}


# ---------------------------------------------------------------------------
# Visualization config
# ---------------------------------------------------------------------------


def load_viz_config(config_path=None) -> dict:
    """Load visualization configuration.

    Resolution order:
    1. ``configs/visualization/`` at the project root.
    2. ``bvr_marl_core/visualization/configs/`` (package fallback).

    Args:
        config_path: Absolute path, relative path, or bare config name.
                     Defaults to ``default.yaml``.
    Returns:
        dict: Configuration dictionary (empty dict if file not found).
    """
    top_level_dir = visualization_configs_root()
    pkg_dir = core_root() / "visualization" / "configs"

    if config_path is None:
        top_level = top_level_dir / "default.yaml"
        pkg_default = pkg_dir / "viz_config_default.yaml"
        full_path = top_level if top_level.exists() else pkg_default
    else:
        p = Path(config_path)
        if p.is_absolute():
            full_path = p
        else:
            stem = p.stem if p.suffix else str(p)
            name = f"{stem}.yaml"
            candidates = [
                top_level_dir / name,
                pkg_dir / name,
                pkg_dir / f"viz_config_{stem}.yaml",
            ]
            full_path = next((c for c in candidates if c.exists()), top_level_dir / name)

    if not full_path.exists():
        return {}
    with open(full_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config else {}


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------


def load_shared_config(name: str) -> dict:
    """Load a shared config file by stem name (e.g. ``"paths"`` or ``"runtime"``).

    Resolution order:
    1. ``configs/shared/<name>.yaml`` at the project root.
    2. ``bvr_marl_core/configs/shared/<name>.yaml`` (package fallback).

    Returns an empty dict if the file does not exist.
    """
    stem = Path(name).stem if Path(name).suffix else name
    filename = f"{stem}.yaml"
    candidates = [
        shared_configs_root() / filename,
        core_root() / "configs" / "shared" / filename,
    ]
    full_path = next((c for c in candidates if c.exists()), shared_configs_root() / filename)
    if not full_path.exists():
        return {}
    with open(full_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config else {}
