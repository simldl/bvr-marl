"""Information-access modes shared by policies and scripted controllers."""

from __future__ import annotations

from enum import StrEnum

from bvr_marl_core.domain.information import TrackSnapshot


class InformationMode(StrEnum):
    """Declare whether a controller receives operational estimates or evaluator truth."""

    SENSOR_LIMITED = "sensor_limited"
    ORACLE = "oracle"


class TruthAccessError(RuntimeError):
    """Raised when evaluator truth crosses a sensor-limited interface."""


def resolve_information_mode(value: object, *, default: InformationMode) -> InformationMode:
    """Normalize configuration values without silently accepting unknown modes."""
    if value is None:
        return default
    if isinstance(value, InformationMode):
        return value
    try:
        return InformationMode(str(value).strip().lower())
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in InformationMode)
        raise ValueError(f"Unknown information mode {value!r}; expected one of: {valid}.") from exc


def reject_truth_handle(value: object, *, boundary: str) -> None:
    """Reject Unit/Missile-like objects at an explicitly sensor-limited boundary.

    Detection is intentionally structural so this module does not import simulator
    entity implementations. Immutable reports, snapshots, arrays, and scalar values
    do not expose the entity-level ``position``/``velocity``/``id`` combination.
    """
    if value is None:
        return
    if all(hasattr(value, attribute) for attribute in ("id", "position", "velocity")):
        raise TruthAccessError(
            f"Evaluator entity {type(value).__name__} crossed sensor-limited boundary {boundary}."
        )


def audit_sensor_limited_simulator(simulator: object) -> None:
    """Fail if a live entity is retained in an operational sensor/weapon record.

    Physics and terminal adjudication still own the authoritative roster. This
    audit deliberately inspects only objects exported to sensor-limited consumers
    or retained by operational missile guidance.
    """
    active_units = getattr(simulator, "active_units", {})
    for unit_id, unit in active_units.items():
        radar = getattr(unit, "radar", None)
        for index, track in enumerate(getattr(radar, "cached_tracks", ()) or ()):
            if not isinstance(track, TrackSnapshot):
                raise TruthAccessError(
                    f"unit[{unit_id}].radar.track[{index}] is not a TrackSnapshot"
                )
        sensor = getattr(unit, "sensor", None)
        for index, track in enumerate(getattr(sensor, "sensor_tracks", ()) or ()):
            if not isinstance(track, TrackSnapshot):
                raise TruthAccessError(
                    f"unit[{unit_id}].sensor.track[{index}] is not a TrackSnapshot"
                )
        if getattr(unit, "is_missile", False):
            reject_truth_handle(
                getattr(unit, "target", None), boundary=f"missile[{unit_id}].target"
            )
