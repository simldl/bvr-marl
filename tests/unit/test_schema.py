"""Unit tests for schema versioning, migration, and model structure.

Verifies:
- Version constants are consistent.
- migrate_config upgrades unversioned, 1.0, and current-version dicts correctly.
- Real field renames introduced in 2.0 are applied by migration.
- Unsupported versions raise ValueError.
- Top-level models (ScenarioConfig, SimulationConfig) carry schema_version.
- Nested models (AgentConfig, TeamConfig, WeaponConfig, SensorConfig) do NOT
  carry schema_version (version is centralized at the document level).
- VersionedModel.from_raw runs migration then validation end-to-end.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bvr_marl_core.schema import (
    SCHEMA_VERSION,
    SUPPORTED_VERSIONS,
    AgentConfig,
    NestedModel,
    ScenarioConfig,
    SensorConfig,
    SimulationConfig,
    TeamConfig,
    VersionedModel,
    WeaponConfig,
    migrate_config,
)

# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------


def test_current_version_is_in_supported():
    assert SCHEMA_VERSION in SUPPORTED_VERSIONS


def test_supported_versions_ordered():
    # Simple sanity: 1.0 comes before 2.0
    assert SUPPORTED_VERSIONS.index("1.0") < SUPPORTED_VERSIONS.index("2.0")


# ---------------------------------------------------------------------------
# migrate_config: unversioned → 1.0 → 2.0
# ---------------------------------------------------------------------------


def test_migrate_unversioned_injects_and_upgrades_to_current():
    raw = {"scenario_id": "s1", "teams": []}
    result = migrate_config(raw)
    assert result["schema_version"] == SCHEMA_VERSION


def test_migrate_1_0_to_2_0_renames_duration_s():
    raw = {"schema_version": "1.0", "scenario_id": "s1", "teams": [], "duration_s": 180.0}
    result = migrate_config(raw)
    assert result["schema_version"] == "2.0"
    assert "max_duration_s" in result
    assert result["max_duration_s"] == 180.0
    assert "duration_s" not in result


def test_migrate_1_0_to_2_0_renames_gun_hit_radius():
    raw = {"schema_version": "1.0", "tick_secs": 0.5, "gun_hit_radius_m": 8.0}
    result = migrate_config(raw)
    assert result["schema_version"] == "2.0"
    assert "gun_lethal_radius_m" in result
    assert result["gun_lethal_radius_m"] == 8.0
    assert "gun_hit_radius_m" not in result


def test_migrate_current_version_is_idempotent():
    raw = {"schema_version": SCHEMA_VERSION, "scenario_id": "s1", "teams": []}
    result = migrate_config(raw)
    assert result["schema_version"] == SCHEMA_VERSION


def test_migrate_unknown_version_raises():
    raw = {"schema_version": "99.0", "scenario_id": "s1", "teams": []}
    with pytest.raises(ValueError, match="Cannot migrate"):
        migrate_config(raw)


# ---------------------------------------------------------------------------
# Centralized schema version: top-level models only
# ---------------------------------------------------------------------------


def test_versioned_model_subclasses_have_schema_version():
    """ScenarioConfig and SimulationConfig must be VersionedModel subclasses."""
    assert issubclass(ScenarioConfig, VersionedModel)
    assert issubclass(SimulationConfig, VersionedModel)


def test_nested_models_are_nested_model_subclasses():
    """Nested config models must inherit from NestedModel, not VersionedModel."""
    for cls in (AgentConfig, TeamConfig, WeaponConfig, SensorConfig):
        assert issubclass(cls, NestedModel), f"{cls.__name__} should be a NestedModel"
        assert not issubclass(cls, VersionedModel), (
            f"{cls.__name__} must not inherit VersionedModel — "
            "schema_version is a top-level document concern only"
        )


def test_nested_model_has_no_schema_version_field():
    """NestedModel subclasses must not expose a schema_version field."""
    for cls in (AgentConfig, TeamConfig, WeaponConfig, SensorConfig):
        assert "schema_version" not in cls.model_fields, (
            f"{cls.__name__} must not declare schema_version"
        )


# ---------------------------------------------------------------------------
# VersionedModel: version gating
# ---------------------------------------------------------------------------


def test_versioned_model_rejects_unsupported_version():
    with pytest.raises((ValidationError, ValueError)):
        SimulationConfig.model_validate({"schema_version": "0.1"})


def test_versioned_model_accepts_supported_versions():
    for ver in SUPPORTED_VERSIONS:
        cfg = SimulationConfig.model_validate({"schema_version": ver})
        assert cfg.schema_version == ver


# ---------------------------------------------------------------------------
# VersionedModel.from_raw end-to-end
# ---------------------------------------------------------------------------


def _minimal_scenario_raw_1_0() -> dict:
    return {
        "schema_version": "1.0",
        "scenario_id": "test_scenario",
        "teams": [
            {
                "team_id": "blue",
                "affiliation": "friendly",
                "agents": [
                    {
                        "agent_id": "a1",
                        "aircraft_type": "f16",
                        "initial_speed_ms": 280.0,
                    }
                ],
            }
        ],
        "duration_s": 120.0,
    }


def test_from_raw_upgrades_1_0_scenario_to_current():
    raw = _minimal_scenario_raw_1_0()
    cfg = ScenarioConfig.from_raw(raw)
    assert cfg.schema_version == SCHEMA_VERSION
    assert cfg.max_duration_s == 120.0
    assert cfg.teams[0].agents[0].initial_speed_ms == 280.0


def test_from_raw_current_version_passes_through():
    raw = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": "s2",
        "teams": [],
    }
    cfg = ScenarioConfig.from_raw(raw)
    assert cfg.scenario_id == "s2"


def test_simulation_config_from_raw_1_0_renames_gun_radius():
    raw = {
        "schema_version": "1.0",
        "gun_hit_radius_m": 7.0,
    }
    cfg = SimulationConfig.from_raw(raw)
    assert cfg.gun_lethal_radius_m == 7.0


# ---------------------------------------------------------------------------
# Chained migration: unversioned → 1.0 → 2.0 in one call
# ---------------------------------------------------------------------------


def test_chained_migration_unversioned_with_both_legacy_fields():
    """A completely pre-versioned dict with legacy fields from multiple hops
    must be fully upgraded in a single migrate_config() call."""
    raw = {
        # No schema_version — treated as pre-1.0
        "scenario_id": "legacy",
        "teams": [],
        "duration_s": 90.0,  # 1.0→2.0 rename
    }
    result = migrate_config(raw)
    assert result["schema_version"] == SCHEMA_VERSION
    assert "max_duration_s" in result
    assert result["max_duration_s"] == 90.0
    assert "duration_s" not in result


def test_chained_migration_unversioned_sim_config():
    """Unversioned SimulationConfig with a legacy gun field upgrades cleanly."""
    raw = {
        "tick_secs": 0.5,
        "gun_hit_radius_m": 12.0,
    }
    result = migrate_config(raw)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["gun_lethal_radius_m"] == 12.0
    assert "gun_hit_radius_m" not in result


def test_chained_migration_preserves_unrecognised_fields():
    """Unknown extra fields are preserved unchanged through migration."""
    raw = {
        "schema_version": "1.0",
        "scenario_id": "s1",
        "teams": [],
        "custom_metadata": {"author": "test"},
    }
    result = migrate_config(raw)
    assert result["custom_metadata"] == {"author": "test"}


# ---------------------------------------------------------------------------
# Failure tests: unsupported future version and structurally invalid inputs
# ---------------------------------------------------------------------------


def test_migrate_future_version_raises():
    """A schema_version ahead of the current version must raise ValueError."""
    raw = {"schema_version": "9.9", "scenario_id": "future", "teams": []}
    with pytest.raises(ValueError, match="Cannot migrate"):
        migrate_config(raw)


def test_migrate_empty_dict_injects_version():
    """An empty dict (no version key) must get schema_version injected and upgraded."""
    result = migrate_config({})
    assert result["schema_version"] == SCHEMA_VERSION


def test_versioned_model_rejects_future_version():
    """VersionedModel must reject a schema_version not in SUPPORTED_VERSIONS."""
    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):
        ScenarioConfig.model_validate(
            {
                "schema_version": "9.9",
                "scenario_id": "s1",
                "teams": [],
            }
        )


# ---------------------------------------------------------------------------
# Schema model: mutable default isolation
# ---------------------------------------------------------------------------


def test_agent_config_sensors_list_is_independent():
    """Two AgentConfig instances must not share the same sensors list."""
    from bvr_marl_core.schema import AgentConfig

    a1 = AgentConfig(agent_id="a1", aircraft_type="f22")
    a2 = AgentConfig(agent_id="a2", aircraft_type="f35")
    assert a1.sensors is not a2.sensors, (
        "AgentConfig.sensors must not share a mutable default between instances"
    )


def test_agent_config_weapons_list_is_independent():
    """Two AgentConfig instances must not share the same weapons list."""
    from bvr_marl_core.schema import AgentConfig

    a1 = AgentConfig(agent_id="a1", aircraft_type="f22")
    a2 = AgentConfig(agent_id="a2", aircraft_type="f35")
    assert a1.weapons is not a2.weapons


def test_agent_config_initial_position_is_independent():
    """Two AgentConfig instances must have independent initial_position_enu lists."""
    from bvr_marl_core.schema import AgentConfig

    a1 = AgentConfig(agent_id="a1", aircraft_type="f22")
    a2 = AgentConfig(agent_id="a2", aircraft_type="f35")
    a1.initial_position_enu[0] = 9999.0
    assert a2.initial_position_enu[0] != 9999.0, (
        "Mutating one AgentConfig.initial_position_enu must not affect another instance"
    )


def test_team_config_agents_list_is_independent():
    """Two TeamConfig instances must not share the same agents list."""
    from bvr_marl_core.schema import TeamConfig

    t1 = TeamConfig(team_id="blue", affiliation="friendly")
    t2 = TeamConfig(team_id="red", affiliation="hostile")
    assert t1.agents is not t2.agents
