"""
Centralized filesystem path resolution for air_to_air_rl.

All code that needs to locate package resources, default configs, or
assets should import helpers from this module rather than building
paths relative to __file__ in individual files.

Usage
-----
    from air_to_air_rl.core.paths import package_root, rl_configs_root

Runtime output locations (checkpoints, logs, models) are intentionally
NOT defined here — those are user-controlled and passed as explicit
arguments to the relevant commands.
"""

from __future__ import annotations

import os
from pathlib import Path


def package_root() -> Path:
    """Absolute path to the installed air_to_air_rl package directory."""
    return Path(__file__).parent.parent


def project_root() -> Path:
    """
    Project root directory (the directory that contains pyproject.toml).

    For an editable install the package lives at <project_root>/src/air_to_air_rl/,
    so this walks two levels up from package_root().  Falls back to the user home
    directory under ~/.air_to_air_rl/ when the package is installed non-editably.
    """
    candidate = package_root().parent.parent  # src/ -> project root
    if (candidate / "pyproject.toml").exists():
        return candidate
    return Path.home() / ".air_to_air_rl"


def runtime_root() -> Path:
    """
    Root directory for all runtime state files.

    Resolved in order:
    1. AIR2AIR_RUNTIME_DIR environment variable (if set).
    2. <project_root>/runtime/

    The directory is created automatically on first access.
    """
    env = os.environ.get("AIR2AIR_RUNTIME_DIR")
    root = Path(env) if env else project_root() / "runtime"
    root.mkdir(parents=True, exist_ok=True)
    return root


def rl_configs_root() -> Path:
    """Default training configuration directory (src/air_to_air_rl/rl/configs/)."""
    return package_root() / "rl" / "configs"


def visualization_root() -> Path:
    """Visualization package directory containing configs and assets."""
    return package_root() / "visualization"


def visualization_config(name: str = "viz_config_default.yaml") -> Path:
    """Path to a named visualization config file inside the package."""
    return visualization_root() / "configs" / name


def tacview_config() -> Path:
    """Default Tacview configuration file (prefers top-level configs/tacview/default.yaml)."""
    top_level = project_root() / "configs" / "tacview" / "default.yaml"
    if top_level.exists():
        return top_level
    return package_root() / "tacview" / "tacview_config.yaml"


def symbols_root() -> Path:
    """Root directory for visualization symbol assets."""
    return visualization_root() / "symbols"


def gui_app() -> Path:
    """Absolute path to the Streamlit GUI application file."""
    return package_root() / "gui" / "app.py"


def gui_config() -> Path:
    """GUI default configuration file (prefers top-level configs/gui/default.yaml)."""
    top_level = project_root() / "configs" / "gui" / "default.yaml"
    if top_level.exists():
        return top_level
    return package_root() / "gui" / "gui_config_default.yaml"


def shared_config(name: str) -> Path:
    """
    Path to a shared config file.

    Prefers top-level configs/shared/<name>.yaml; falls back to package
    src/air_to_air_rl/configs/shared/<name>.yaml if the top-level file
    does not exist.
    """
    top_level = project_root() / "configs" / "shared" / name
    if not top_level.suffix:
        top_level = top_level.with_suffix(".yaml")
    if top_level.exists():
        return top_level
    pkg = package_root() / "configs" / "shared" / top_level.name
    return pkg if pkg.exists() else top_level
