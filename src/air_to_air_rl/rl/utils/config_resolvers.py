"""
Configuration resolvers for BVR environment.

Handles resolution of missile types and other configurable parameters
from YAML configuration to actual class instances.
"""

from typing import Any, Optional

from air_to_air_rl.rl.utils.type_maps import MISSILE_TYPE_MAP


def resolve_missile_config(missile_config: dict[str, Any] | None) -> dict[str, list[type]]:
    """
    Resolve missile configuration from string names to actual missile classes.

    Args:
        missile_config: Dictionary with 'agent_missiles' and 'opponent_missiles' keys,
                       each containing a list of missile type strings.

    Returns:
        Dictionary with resolved missile classes for agents and opponents.
        Returns empty lists if no config provided.

    Raises:
        ValueError: If a missile type string is not recognized.
    """
    if missile_config is None:
        return {
            "agent_missiles": [],
            "opponent_missiles": [],
        }

    agent_missile_names = missile_config.get("agent_missiles", [])
    opponent_missile_names = missile_config.get("opponent_missiles", [])

    agent_missiles = []
    for name in agent_missile_names:
        if name not in MISSILE_TYPE_MAP:
            raise ValueError(
                f"Unknown missile type: {name}. Available: {list(MISSILE_TYPE_MAP.keys())}"
            )
        agent_missiles.append(MISSILE_TYPE_MAP[name])

    opponent_missiles = []
    for name in opponent_missile_names:
        if name not in MISSILE_TYPE_MAP:
            raise ValueError(
                f"Unknown missile type: {name}. Available: {list(MISSILE_TYPE_MAP.keys())}"
            )
        opponent_missiles.append(MISSILE_TYPE_MAP[name])

    return {
        "agent_missiles": agent_missiles,
        "opponent_missiles": opponent_missiles,
    }


def resolve_loadout_config(loadout_config: dict[str, Any] | None) -> dict[str, Any]:
    """
    Resolve loadout configuration overrides.

    Args:
        loadout_config: Dictionary with optional overrides for max_missiles,
                       flares, chaff, ecm, decoys.

    Returns:
        Dictionary with resolved loadout values (None for non-overridden values).
    """
    if loadout_config is None:
        return {}

    return {
        "max_missiles": loadout_config.get("max_missiles"),
        "flares": loadout_config.get("flares"),
        "chaff": loadout_config.get("chaff"),
        "ecm": loadout_config.get("ecm"),
        "decoys": loadout_config.get("decoys"),
    }


def resolve_datalink_config(config: dict[str, Any]) -> str:
    """
    Resolve datalink mode from configuration.

    Args:
        config: Environment configuration dictionary.

    Returns:
        Datalink mode string: "full", "own", "none", "other", or "msl_support".
        Defaults to "full" if not specified.
    """
    return config.get("datalink_mode", "full")
