"""
Path helpers for bvr_marl_core.

Provides functions to locate core-owned resources (configs, GUI assets,
Tacview outputs, runtime state).  Extension-package resources are resolved
by that package's own path helpers.

Usage
-----
    from bvr_marl_core.utils.paths import (
        core_project_root,
        core_runtime_root,
        rl_configs_root,
    )
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Package and repo roots
# ---------------------------------------------------------------------------


def core_root() -> Path:
    """Absolute path to the installed bvr_marl_core package directory."""
    return Path(__file__).parent.parent


def core_project_root() -> Path:
    """Root of the bvr-marl-core repository (contains pyproject.toml).

    For an editable install the package lives at
    ``<repo_root>/src/bvr_marl_core/``, so this walks two levels up from
    ``core_root()``.  Falls back to ``~/.bvr_marl_core/`` when the
    package is installed non-editably and the repo layout is not present.
    """
    candidate = core_root().parent.parent  # src/ -> repo root
    if (candidate / "pyproject.toml").exists():
        return candidate
    return Path.home() / ".bvr_marl_core"


def sibling_model_roots() -> list[Path]:
    """Additional model roots contributed by sibling extension repositories.

    Extension packages are commonly checked out next to bvr-marl-core.  Any
    sibling directory that contains a ``models/`` subdirectory is treated as an
    extra model root so GUI checkpoint pickers can discover their runs.  Extra
    roots may also be supplied explicitly via the ``BVR_MODEL_ROOTS``
    environment variable (``os.pathsep``-separated paths), which takes
    precedence and is always included when set.
    """
    roots: list[Path] = []

    env = os.environ.get("BVR_MODEL_ROOTS")
    if env:
        roots.extend(Path(part) for part in env.split(os.pathsep) if part)

    try:
        repo = core_project_root().resolve()
        for sibling in repo.parent.iterdir():
            if not sibling.is_dir() or sibling.resolve() == repo:
                continue
            candidate = sibling / "models"
            if candidate.is_dir():
                roots.append(candidate)
    except Exception:
        pass

    return roots


def core_runtime_root() -> Path:
    """Root directory for all core runtime state files.

    Resolved in order:
    1. ``BVR_RUNTIME_DIR`` environment variable (if set).
    2. ``AIR2AIR_RUNTIME_DIR`` environment variable (legacy, if set).
    3. ``<core_project_root>/runtime/``

    The directory is created automatically on first access.
    """
    env = os.environ.get("BVR_RUNTIME_DIR") or os.environ.get("AIR2AIR_RUNTIME_DIR")
    root = Path(env) if env else core_project_root() / "runtime"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Config roots — all resolve within the core repo
# ---------------------------------------------------------------------------


def training_configs_root() -> Path:
    """Top-level training config directory (``configs/training/``)."""
    return core_project_root() / "configs" / "training"


def visualization_configs_root() -> Path:
    """Top-level visualization config directory (``configs/visualization/``)."""
    return core_project_root() / "configs" / "visualization"


def shared_configs_root() -> Path:
    """Top-level shared config directory (``configs/shared/``)."""
    return core_project_root() / "configs" / "shared"


def gui_configs_root() -> Path:
    """Top-level GUI config directory (``configs/gui/``)."""
    return core_project_root() / "configs" / "gui"


# ---------------------------------------------------------------------------
# Asset / output roots
# ---------------------------------------------------------------------------


def models_root() -> Path:
    """Root directory for trained model checkpoints.

    Resolution order:
    1. ``BVR_MODELS_DIR`` environment variable.
    2. ``<core_project_root>/models/``
    """
    env = os.environ.get("BVR_MODELS_DIR")
    return Path(env) if env else core_project_root() / "models"


def analysis_output_root() -> Path:
    """Root directory for analysis outputs (plots, reports)."""
    return core_project_root() / "analysis_output"


def exported_plots_root() -> Path:
    """Default directory for exported analysis figures.

    This is the single source of truth shared by the GUI's Export-Plots tab and
    the "Analysis Output Information" panel, so the path shown always matches the
    path written to.

    Resolution order:
    1. ``BVR_EXPORTED_PLOTS_DIR`` environment variable.
    2. ``<workspace_root>/exported_plots/`` — the sibling of the repo root
       (e.g. ``Code/exported_plots`` when the repo lives at ``Code/bvr-marl-core``).
    """
    env = os.environ.get("BVR_EXPORTED_PLOTS_DIR")
    return Path(env) if env else core_project_root().parent / "exported_plots"


def tacview_output_root() -> Path:
    """Root directory for generated Tacview ACMI files.

    Resolution order:
    1. ``BVR_TACVIEW_OUTPUT_DIR`` environment variable.
    2. ``<core_project_root>/tacview/logs/``
    """
    env = os.environ.get("BVR_TACVIEW_OUTPUT_DIR")
    return Path(env) if env else core_project_root() / "tacview" / "logs"


# ---------------------------------------------------------------------------
# Named config file paths
# ---------------------------------------------------------------------------


def rl_configs_root() -> Path:
    """Optional legacy package training-config directory.

    Resolution order:
    1. ``BVR_RL_CONFIGS_DIR`` environment variable.
    2. ``bvr_marl_core/rl/configs/`` relative to this package, when present.

    Active checked-in training defaults live in the project-level
    ``configs/training`` directory.
    """
    env = os.environ.get("BVR_RL_CONFIGS_DIR")
    if env:
        return Path(env)
    return core_root() / "rl" / "configs"


def gui_app() -> Path:
    """Absolute path to the Streamlit GUI application file."""
    env = os.environ.get("BVR_GUI_APP")
    if env:
        return Path(env)
    return core_root() / "gui" / "app.py"


def gui_config() -> Path:
    """GUI default configuration file.

    Resolution order:
    1. ``BVR_GUI_CONFIG`` environment variable.
    2. ``<core_project_root>/configs/gui/default.yaml``
    3. ``bvr_marl_core/gui/gui_config_default.yaml`` (package fallback).
    """
    env = os.environ.get("BVR_GUI_CONFIG")
    if env:
        return Path(env)
    top_level = core_project_root() / "configs" / "gui" / "default.yaml"
    if top_level.exists():
        return top_level
    return core_root() / "gui" / "gui_config_default.yaml"


def tacview_config() -> Path:
    """Canonical Tacview configuration file path.

    Resolution order:
    1. ``<core_project_root>/configs/tacview/default.yaml``
    2. ``bvr_marl_core/tacview/tacview_config.yaml`` (package fallback).
    """
    top_level = core_project_root() / "configs" / "tacview" / "default.yaml"
    if top_level.exists():
        return top_level
    return core_root() / "tacview" / "tacview_config.yaml"
