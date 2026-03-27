"""Configuration loading utilities."""

from pathlib import Path

import yaml

from air_to_air_rl.core.paths import (
    gui_config as _gui_config_path,
)
from air_to_air_rl.core.paths import (
    project_root,
    rl_configs_root,
    visualization_root,
)
from air_to_air_rl.core.paths import (
    shared_config as _shared_config_path,
)


def _resolve_config(
    name_or_path: str | Path,
    default_dir: Path,
    top_level_subdir: str,
) -> Path:
    """
    Resolve a config name or path to an absolute Path.

    Resolution order:
    1. Absolute path  → used as-is.
    2. Path with separator → resolved relative to project_root().
    3. Bare filename or stem → searched in:
         a. <project_root>/configs/<top_level_subdir>/  (top-level project configs)
         b. default_dir (package-installed defaults)
         c. default_dir with legacy "train_config_<stem>" name pattern
    """
    p = Path(name_or_path)

    if p.is_absolute():
        return p

    if "/" in str(name_or_path) or "\\" in str(name_or_path):
        return project_root() / p

    # Bare name — try adding .yaml if missing
    stem = p.stem if p.suffix else str(p)
    candidates = [
        project_root() / "configs" / top_level_subdir / f"{stem}.yaml",
        default_dir / f"{stem}.yaml",
        default_dir / f"train_config_{stem}.yaml",  # legacy name fallback
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Return the package default path (will raise FileNotFoundError if missing)
    return default_dir / f"{stem}.yaml"


def resolve_training_config(name_or_path: "str | Path") -> Path:
    """Resolve a training config name/path to an absolute Path (file may not exist)."""
    return _resolve_config(name_or_path, rl_configs_root(), "training")


def load_train_config(config_path=None) -> dict:
    """
    Load training configuration.

    Args:
        config_path: Absolute path, relative path, or bare config name.
                     Defaults to base.yaml in the package training configs.
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        default = rl_configs_root() / "train_config.yaml"
        # Prefer top-level configs/training/base.yaml if it exists
        top_level = project_root() / "configs" / "training" / "base.yaml"
        full_path = top_level if top_level.exists() else default
    else:
        full_path = _resolve_config(config_path, rl_configs_root(), "training")

    with open(full_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_viz_config(config_path=None) -> dict:
    """
    Load visualization configuration.

    Args:
        config_path: Absolute path, relative path, or bare config name.
                     Defaults to default.yaml in the package visualization configs.
    Returns:
        dict: Configuration dictionary (empty dict if file not found)
    """
    viz_configs_dir = visualization_root() / "configs"

    if config_path is None:
        # Prefer top-level configs/visualization/default.yaml
        top_level = project_root() / "configs" / "visualization" / "default.yaml"
        pkg_default = viz_configs_dir / "viz_config_default.yaml"
        full_path = top_level if top_level.exists() else pkg_default
    else:
        full_path = _resolve_config(config_path, viz_configs_dir, "visualization")

    if not full_path.exists():
        print(f"Warning: Visualization config not found: {full_path}")
        return {}

    with open(full_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config else {}


def load_symbol_config() -> dict:
    """Load symbol configuration for visualization."""
    return load_viz_config("symbol_config.yaml")


def load_gui_config() -> dict:
    """
    Load GUI default configuration.

    Searches configs/gui/default.yaml first; falls back to the package default.
    Returns an empty dict if neither file exists.
    """
    full_path = _gui_config_path()
    if not full_path.exists():
        return {}
    with open(full_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config else {}


def load_shared_config(name: str) -> dict:
    """
    Load a shared config file by stem name (e.g. "paths" or "runtime").

    Searches configs/shared/<name>.yaml first; falls back to the package default.
    Returns an empty dict if the file does not exist.
    """
    full_path = _shared_config_path(name)
    if not full_path.exists():
        return {}
    with open(full_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config else {}


def resolve_relative_path(path) -> Path:
    """
    Resolve a path to absolute.

    Absolute paths are returned unchanged.
    Relative paths are resolved relative to CWD — this is intentional:
    this function is called with user-supplied paths (e.g. from CLI flags)
    where resolving against CWD matches the user's expectation.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    # Explicit user input — resolve relative to CWD as the user intends.
    return p.resolve()
