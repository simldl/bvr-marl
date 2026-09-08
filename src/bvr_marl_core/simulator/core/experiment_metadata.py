"""Canonical experiment metadata used by replays, checkpoints, and reports."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

METADATA_SCHEMA_VERSION = "1.0"
OBSERVATION_SCHEMA_VERSION = "2.1"
ACTION_SCHEMA_VERSION = "2.0"
MODEL_STATUS_MATRIX_VERSION = "2026-07-21.1"
NETWORK_PICTURE_VERSION = "1.0"
PLATFORM_PARAMETER_SET_VERSION = "synthetic-2026-07"
WEAPON_PARAMETER_SET_VERSION = "synthetic-2026-07"
VALIDATION_SUITE_VERSION = "2026-07-21.1"


def canonical_hash(value: Mapping[str, Any] | None) -> str:
    """Return a stable SHA-256 for JSON-compatible configuration data."""
    payload = json.dumps(value or {}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_commit(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _dependency_fingerprint() -> str:
    distributions = sorted(
        f"{dist.metadata.get('Name', '')}=={dist.version}"
        for dist in importlib.metadata.distributions()
    )
    return hashlib.sha256("\n".join(distributions).encode("utf-8")).hexdigest()


@lru_cache(maxsize=4)
def _static_metadata(root: Path) -> dict[str, Any]:
    """Cache process-stable provenance that is expensive to enumerate.

    Dependency enumeration and two git subprocesses used to run for every
    Simulator construction/reset, which is material in short RL episodes and
    population-based workloads. A new process naturally refreshes the values.
    """
    # An extension package records its own provenance by pointing BVR_EXTENSION_ROOT
    # at its repository root; core does not know or assume where that is.
    extension_root = os.environ.get("BVR_EXTENSION_ROOT", "").strip()
    extension_path = Path(extension_root) if extension_root else None
    return {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "core_commit": _git_commit(root),
        "extension_commit": (
            _git_commit(extension_path)
            if extension_path is not None and extension_path.is_dir()
            else "unknown"
        ),
        "python_version": platform.python_version(),
        "dependency_fingerprint": _dependency_fingerprint(),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "model_status_matrix_version": MODEL_STATUS_MATRIX_VERSION,
        "network_picture_version": NETWORK_PICTURE_VERSION,
        "platform_parameter_set_version": PLATFORM_PARAMETER_SET_VERSION,
        "weapon_parameter_set_version": WEAPON_PARAMETER_SET_VERSION,
        "validation_suite_version": VALIDATION_SUITE_VERSION,
    }


def build_experiment_metadata(
    config: Mapping[str, Any] | None = None, *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Build the reproducibility contract without requiring an installed package."""
    root = (repository_root or Path(__file__).resolve().parents[4]).resolve()
    metadata = dict(_static_metadata(root))
    metadata["configuration_hash"] = canonical_hash(config)
    return metadata


def prototype_warnings(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Return explicit warnings for privileged or non-production experiment modes."""
    warnings: list[str] = []
    if metadata.get("information_mode") == "oracle":
        warnings.append("oracle information mode is enabled")
    if metadata.get("reward_information_mode") == "privileged_training":
        warnings.append("privileged training rewards are enabled")
    return tuple(warnings)
