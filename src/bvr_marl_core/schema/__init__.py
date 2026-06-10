"""Canonical versioned config/scenario schema for bvr_marl_core."""

from .base import NestedModel, VersionedModel
from .migration import migrate_config
from .scenario import (
    AgentConfig,
    ScenarioConfig,
    SensorConfig,
    TeamConfig,
    WeaponConfig,
)
from .simulation import SimulationConfig
from .validation import validate_config
from .version import SCHEMA_VERSION, SUPPORTED_VERSIONS

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_VERSIONS",
    "NestedModel",
    "VersionedModel",
    "ScenarioConfig",
    "SimulationConfig",
    "AgentConfig",
    "TeamConfig",
    "WeaponConfig",
    "SensorConfig",
    "validate_config",
    "migrate_config",
]
