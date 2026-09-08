"""Centralized schema migration pipeline.

All raw config/scenario dicts loaded from disk must pass through
``migrate_config`` before being handed to a VersionedModel.  This function
upgrades the dict step-by-step from its declared version to SCHEMA_VERSION.

Adding a new schema version
---------------------------
1. Add the new version string to ``SUPPORTED_VERSIONS`` in version.py.
2. Add a corresponding entry to ``VERSION_CHANGELOG`` in version.py.
3. Implement ``_migrate_X_to_Y(raw)`` below and register it in
   ``_MIGRATION_STEPS``.

The migration steps are applied in order, so a dict at version "1.0" that
needs to reach "2.0" and then "3.0" will run _migrate_1_to_2 followed by
_migrate_2_to_3.
"""

from __future__ import annotations

from bvr_marl_core.schema.version import SCHEMA_VERSION, SUPPORTED_VERSIONS

# ---------------------------------------------------------------------------
# Step implementations — add one function per version hop
# ---------------------------------------------------------------------------


def _migrate_unversioned_to_1_0(raw: dict) -> dict:
    """Inject schema_version into configs that predate versioning."""
    raw = dict(raw)
    raw["schema_version"] = "1.0"
    return raw


def _migrate_1_0_to_2_0(raw: dict) -> dict:
    """Migrate config dicts from schema 1.0 to 2.0.

    Changes applied:
    - ScenarioConfig: ``duration_s`` renamed to ``max_duration_s``.
    - SimulationConfig: ``gun_hit_radius_m`` renamed to ``gun_lethal_radius_m``.
    """
    raw = dict(raw)
    # ScenarioConfig: rename duration_s -> max_duration_s
    if "duration_s" in raw:
        raw["max_duration_s"] = raw.pop("duration_s")
    # SimulationConfig: rename gun_hit_radius_m -> gun_lethal_radius_m
    if "gun_hit_radius_m" in raw:
        raw["gun_lethal_radius_m"] = raw.pop("gun_hit_radius_m")
    raw["schema_version"] = "2.0"
    return raw


# ---------------------------------------------------------------------------
# Step registry — ordered list of (from_version_or_None, to_version, fn)
# ---------------------------------------------------------------------------

_MIGRATION_STEPS: list[tuple[str | None, str, callable]] = [
    # (None, "1.0") means: apply when schema_version is absent
    (None, "1.0", _migrate_unversioned_to_1_0),
    ("1.0", "2.0", _migrate_1_0_to_2_0),
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def migrate_config(raw: dict) -> dict:
    """Upgrade *raw* to the current SCHEMA_VERSION.

    Steps:
      1. If ``schema_version`` is absent, treat the dict as pre-versioned and
         inject the earliest known version.
      2. Walk the migration step chain until SCHEMA_VERSION is reached.
      3. Raise ValueError for any version that is not in SUPPORTED_VERSIONS
         and has no migration path.

    Returns the migrated dict (the input is not mutated).

    Raises:
        ValueError: If the version is unknown and cannot be migrated.
    """
    raw = dict(raw)
    version = raw.get("schema_version", None)

    # Apply applicable migration steps in sequence
    for from_ver, to_ver, step_fn in _MIGRATION_STEPS:
        current = raw.get("schema_version", None)
        if current == from_ver:
            raw = step_fn(raw)

    final_version = raw.get("schema_version", None)
    if final_version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Cannot migrate schema from version {version!r} to {SCHEMA_VERSION!r}. "
            f"No migration path found. Supported versions: {SUPPORTED_VERSIONS}."
        )

    return raw
