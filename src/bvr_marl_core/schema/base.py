"""Base models for all config/scenario schemas.

There are two base classes:

``NestedModel``
    For models that are always embedded inside a top-level document (e.g.
    AgentConfig inside ScenarioConfig).  These carry no schema_version field
    because their version is governed entirely by the enclosing document.

``VersionedModel``
    For top-level document models that are loaded from disk and therefore need
    explicit version tracking.  This provides:

    * a mandatory ``schema_version`` field defaulting to the current version
    * a single centralized version check that rejects unsupported versions
    * a class-method factory that runs migration before validation

Only top-level document models (ScenarioConfig, SimulationConfig) should
inherit from ``VersionedModel``.  All nested sub-models should inherit from
``NestedModel``.

Usage::

    class MyConfig(VersionedModel):
        some_field: str

    # From a raw dict (migrates then validates):
    cfg = MyConfig.from_raw({"some_field": "value"})

    # From an already-validated dict at current version:
    cfg = MyConfig.model_validate({"schema_version": "2.0", "some_field": "value"})
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from bvr_marl_core.schema.version import SCHEMA_VERSION, SUPPORTED_VERSIONS


class NestedModel(BaseModel):
    """Base class for config models that are always nested inside a top-level document.

    Carries no schema_version field.  Version governance is handled entirely
    by the enclosing VersionedModel before nested dicts are validated.
    """


class VersionedModel(BaseModel):
    """Base class for top-level persisted config documents.

    Applies a centralized version check so that all document-level models
    (ScenarioConfig, SimulationConfig) reject unsupported schema versions
    in one place.  Subclasses must NOT redeclare a ``_check_version``
    validator — the one defined here already applies to every derived class.
    """

    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="before")
    @classmethod
    def _check_schema_version(cls, values):
        """Reject configs whose schema_version is not in SUPPORTED_VERSIONS."""
        if not isinstance(values, dict):
            return values
        version = values.get("schema_version", SCHEMA_VERSION)
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"{cls.__name__}: unsupported schema_version {version!r}. "
                f"Supported versions: {SUPPORTED_VERSIONS}. "
                f"Call migrate_config() before constructing this model."
            )
        return values

    @classmethod
    def from_raw(cls, raw: dict) -> VersionedModel:
        """Migrate *raw* to the current schema version, then validate.

        This is the preferred entry point when loading persisted data.
        It runs migration first so callers do not need to remember to call
        ``migrate_config`` separately.
        """
        from bvr_marl_core.schema.migration import migrate_config

        migrated = migrate_config(raw)
        return cls.model_validate(migrated)
