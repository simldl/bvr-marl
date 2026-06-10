"""Simple YAML-based configuration utilities to replace Hydra."""

from pathlib import Path
from typing import Any, Union

import yaml


class SimpleConfig:
    """Simple configuration class that loads YAML files and supports parameter overrides."""

    def __init__(self, config_dict: dict[str, Any]):
        self._config = config_dict

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with optional default."""
        return self._config.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return the underlying dictionary."""
        return self._config

    def update(self, other: dict[str, Any]) -> None:
        """Update config with another dictionary."""
        self._config.update(other)


def load_config(config_path: Union[str, Path]) -> SimpleConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        SimpleConfig object
    """
    config_path = Path(config_path)

    if not config_path.is_absolute():
        # Resolve relative to project root if not absolute
        project_root = Path(__file__).resolve().parents[1]  # Go up from utils/
        config_path = project_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return SimpleConfig(config_dict)


def apply_overrides(config: SimpleConfig, overrides: list[str]) -> None:
    """
    Apply command-line parameter overrides to the configuration.

    Args:
        config: Configuration object to modify
        overrides: list of override strings in format "key=value" or "nested.key=value"
    """
    for override in overrides:
        if "=" not in override:
            print(f"Warning: Invalid override format '{override}', should be 'key=value'")
            continue

        key_path, value_str = override.split("=", 1)

        # Convert string value to appropriate type
        value = parse_value(value_str)

        # Apply nested key path
        set_nested_key(config._config, key_path, value)


def parse_value(value_str: str) -> Any:
    """
    Parse a string value to appropriate Python type.

    Args:
        value_str: String representation of the value

    Returns:
        Parsed value (int, float, bool, or string)
    """
    value_str = value_str.strip()

    # Boolean values
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"

    # None/null values
    if value_str.lower() in ("none", "null"):
        return None

    # Try to parse as int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try to parse as float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Return as string
    return value_str


def set_nested_key(config_dict: dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set a nested key in the configuration dictionary.

    Args:
        config_dict: Configuration dictionary to modify
        key_path: Dot-separated key path (e.g., "training.steps")
        value: Value to set
    """
    keys = key_path.split(".")
    current = config_dict

    # Navigate to the nested location
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set the final value
    current[keys[-1]] = value


def save_config(config: SimpleConfig, output_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration to save
        output_path: Path to save the configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
