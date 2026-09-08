"""Canonical versioned config/scenario schema for bvr_marl_core."""

from bvr_marl_core.schema.base import NestedModel, VersionedModel
from bvr_marl_core.schema.migration import migrate_config
from bvr_marl_core.schema.scenario import (
    AgentConfig,
    ScenarioConfig,
    SensorConfig,
    TeamConfig,
    WeaponConfig,
)
from bvr_marl_core.schema.simulation import SimulationConfig
from bvr_marl_core.schema.validation import validate_config
from bvr_marl_core.schema.version import SCHEMA_VERSION, SUPPORTED_VERSIONS

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
