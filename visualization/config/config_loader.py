"""Configuration loading utilities."""
import os
import yaml
from pathlib import Path
from visualization.utils.path_utils import resolve_relative_path


def load_train_config(config_path=None):
    """
    Load configuration from train_config.yaml.

    Args:
        config_path: Path to config file (optional)

    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "reinforcement_learning" / "configs" / "train_config.yaml"
    else:
        # Resolve relative paths
        config_path = resolve_relative_path(config_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def load_viz_config(config_path=None):
    """
    Load visualization configuration from viz_config.yaml.

    Args:
        config_path: Path to config file (optional)

    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "viz_config.yaml"

    if not os.path.exists(config_path):
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config if config else {}
