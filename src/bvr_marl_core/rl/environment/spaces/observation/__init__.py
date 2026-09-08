"""
Observation Space Module - Modular observation building.

This module provides a modular approach to building observations, with
specialized builders for different observation components.

Main exports:
- ObservationBuilder: Main orchestrator (use this for backward compatibility)
"""

_LAZY_EXPORTS = {
    "ObservationBuilder": ".builder",
    "EnemyInfoBuilder": ".enemy_info_builder",
    "FriendlyInfoBuilder": ".friendly_info_builder",
    "MissileWarningBuilder": ".missile_warning_builder",
    "OwnStateBuilder": ".own_state_builder",
    "PassiveRadarBuilder": ".passive_radar_builder",
    "SimplifiedObservationBuilder": ".simplified_obs_builder",
}

__all__ = [
    "ObservationBuilder",
    "OwnStateBuilder",
    "FriendlyInfoBuilder",
    "EnemyInfoBuilder",
    "MissileWarningBuilder",
    "PassiveRadarBuilder",
    "SimplifiedObservationBuilder",
]


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
