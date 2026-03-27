"""
Config discovery service — shared configuration file discovery for GUI and CLI.

Thin wrappers around ``utils.config_loader`` that expose a consistent
interface for training, visualization, GUI, and shared config discovery.
"""

from __future__ import annotations

from pathlib import Path


def list_training_configs() -> list[str]:
    """Return sorted training config filenames.

    Delegates to :func:`air_to_air_rl.services.training.list_training_configs`
    so there is a single source of truth.
    """
    from air_to_air_rl.services.training import list_training_configs as _list

    return _list()


def list_visualization_configs() -> list[str]:
    """Return sorted visualization config filenames.

    Searches (in order):
    1. ``configs/visualization/`` at the project working directory
    2. The package ``visualization/configs/`` directory
    """
    from air_to_air_rl.core.paths import project_root, visualization_root

    seen: set[str] = set()
    results: list[str] = []

    dirs = [
        project_root() / "configs" / "visualization",
        visualization_root() / "configs",
    ]
    for d in dirs:
        if d.is_dir():
            for ext in ("*.yaml", "*.yml"):
                for f in sorted(d.glob(ext)):
                    if f.name not in seen:
                        seen.add(f.name)
                        results.append(f.name)

    return results


def resolve_training_config(name_or_path: str | Path) -> Path:
    """Resolve a training config name or path to an absolute :class:`Path`."""
    from air_to_air_rl.utils.config_loader import resolve_training_config as _r

    return _r(name_or_path)


def load_training_config(name_or_path: str | Path | None = None) -> dict:
    """Load and return a training config as a plain dict."""
    from air_to_air_rl.utils.config_loader import load_train_config

    return load_train_config(name_or_path)


def load_visualization_config(name_or_path: str | Path | None = None) -> dict:
    """Load and return a visualization config as a plain dict."""
    from air_to_air_rl.utils.config_loader import load_viz_config

    return load_viz_config(name_or_path)


def load_gui_config() -> dict:
    """Load and return the GUI default config (configs/gui/default.yaml)."""
    from air_to_air_rl.utils.config_loader import load_gui_config as _load

    return _load()


def load_shared_config(name: str) -> dict:
    """Load and return a shared config by stem name ('paths' or 'runtime')."""
    from air_to_air_rl.utils.config_loader import load_shared_config as _load

    return _load(name)
