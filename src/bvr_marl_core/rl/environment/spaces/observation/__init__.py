"""
Observation Space Module - Modular observation building.

This module provides a modular approach to building observations, with
specialized builders for different observation components.

Main exports:
- ObservationBuilder: Main orchestrator (use this for backward compatibility)
"""

from .builder import ObservationBuilder
from .enemy_info_builder import EnemyInfoBuilder
from .friendly_info_builder import FriendlyInfoBuilder
from .missile_warning_builder import MissileWarningBuilder
from .own_state_builder import OwnStateBuilder
from .passive_radar_builder import PassiveRadarBuilder
from .simplified_obs_builder import SimplifiedObservationBuilder

__all__ = [
    "ObservationBuilder",
    "OwnStateBuilder",
    "FriendlyInfoBuilder",
    "EnemyInfoBuilder",
    "MissileWarningBuilder",
    "PassiveRadarBuilder",
    "SimplifiedObservationBuilder",
]
