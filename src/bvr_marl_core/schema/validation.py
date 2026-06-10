"""Config validation helpers."""

from __future__ import annotations

from pydantic import BaseModel

from .migration import migrate_config


def validate_config(raw: dict, model_class) -> BaseModel:
    """Migrate and validate a raw dict against a pydantic model.

    Raises ValidationError on failure.
    """
    migrated = migrate_config(raw)
    return model_class.model_validate(migrated)
