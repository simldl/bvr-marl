"""Immutable operational information records.

Production status: foundational API.  These records deliberately contain no
``Unit`` handles or evaluator truth identifiers.  Operational sensor, tracking,
controller, and weapon code should exchange these values across subsystem
boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

import numpy as np

from bvr_marl_core.domain.classification import (
    CLASSIFICATION_LABELS,
    CLASSIFICATION_SCHEMA_VERSION,
)

Scalar = float | int | bool | str
SensorId = int | str
ReportId = int
ReportLineage = tuple[tuple[SensorId, ReportId], ...]


def canonical_report_lineage(values: object) -> ReportLineage:
    """Return de-duplicated, source-qualified report identities in stable order."""
    if values is None:
        return ()
    unique: set[tuple[SensorId, ReportId]] = set()
    for value in values:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("Report lineage entries must be (sensor_id, report_id) pairs.")
        source_id, report_id = value
        if not isinstance(source_id, (int, str)) or not isinstance(report_id, int):
            raise ValueError("Report lineage requires an int/str sensor ID and integer report ID.")
        unique.add((source_id, report_id))
    return tuple(sorted(unique, key=lambda item: (str(item[0]), item[1])))


class SensorType(StrEnum):
    RADAR = "radar"
    IRST = "irst"
    MISSILE_WARNING = "missile_warning"
    PASSIVE_RF = "passive_rf"
    DATALINK = "datalink"


class TrackLifecycle(StrEnum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    REACQUIRED = "reacquired"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class FrameReference:
    """Geodetic origin of an ENU measurement frame."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float


def _tuple(values, expected: int | None = None) -> tuple[float, ...]:
    result = tuple(map(float, values))
    if expected is not None and len(result) != expected:
        raise ValueError(f"Expected {expected} values, received {len(result)}.")
    if not all(np.isfinite(result)):
        raise ValueError("Information records require finite numeric values.")
    return result


def _covariance(values, dimension: int) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(values, dtype=float)
    if array.shape != (dimension, dimension):
        raise ValueError(f"Expected a {dimension}x{dimension} covariance, received {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Covariance contains non-finite values.")
    scale = max(1.0, float(np.max(np.abs(array))))
    # np.allclose is disproportionately slow here (temporaries + nan handling); the
    # values are already known finite, so apply its tolerance formula directly.
    if np.any(np.abs(array - array.T) > 1e-12 * scale + 1e-12 * np.abs(array.T)):
        raise ValueError("Covariance must be symmetric.")
    array = 0.5 * (array + array.T)
    # Fast path: a strictly positive-definite covariance (the overwhelming common
    # case) passes Cholesky, which is ~2x cheaper than the eigendecomposition and
    # proves PSD outright, so no eigenvalue projection can be needed. Only the rare
    # non-PD matrix falls back to eigh.
    try:
        np.linalg.cholesky(array)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(array)
        if float(eigenvalues[0]) < -1e-10 * scale:
            raise ValueError("Covariance must be positive semidefinite.") from None
        if eigenvalues[0] < 0.0:
            # Rotations and Kalman updates can leave round-off-scale negative
            # eigenvalues in an otherwise valid covariance. Project only that
            # numerical residue back onto the PSD cone at the immutable boundary.
            array = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
            array = 0.5 * (array + array.T)
    # ndarray.tolist performs scalar conversion in C; wrapping its six rows as
    # tuples preserves the immutable public contract without nested Python loops.
    return tuple(map(tuple, array.tolist()))


def _metadata(values: Mapping[str, Scalar] | None) -> Mapping[str, Scalar]:
    return MappingProxyType(dict(values or {}))


@dataclass(frozen=True, slots=True)
class SensorReport(Mapping[str, object]):
    """A time-stamped sensor measurement with no live truth references."""

    report_id: int
    source_id: int | str
    acquisition_time_s: float
    measurement: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    frame: FrameReference
    sensor_type: SensorType
    classification_schema_version: int = CLASSIFICATION_SCHEMA_VERSION
    classification_probabilities: tuple[float, ...] | None = None
    metadata: Mapping[str, Scalar] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def lineage(self) -> ReportLineage:
        """Canonical globally unambiguous identity of this report."""
        return ((self.source_id, self.report_id),)

    def __post_init__(self) -> None:
        measurement = _tuple(self.measurement)
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "covariance", _covariance(self.covariance, len(measurement)))
        if self.classification_probabilities is not None:
            probabilities = _tuple(self.classification_probabilities)
            if len(probabilities) != len(CLASSIFICATION_LABELS):
                raise ValueError(
                    f"Classification schema v{CLASSIFICATION_SCHEMA_VERSION} requires "
                    f"{len(CLASSIFICATION_LABELS)} probabilities."
                )
            if any(value < 0.0 for value in probabilities) or not np.isclose(
                sum(probabilities), 1.0
            ):
                raise ValueError("Classification probabilities must be nonnegative and sum to 1.")
            object.__setattr__(self, "classification_probabilities", probabilities)
        if self.classification_schema_version != CLASSIFICATION_SCHEMA_VERSION:
            raise ValueError("Unsupported classification_schema_version.")
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def measurement_array(self) -> np.ndarray:
        """Return a writable copy for numerical filtering."""
        return np.asarray(self.measurement, dtype=float).copy()

    def covariance_array(self) -> np.ndarray:
        """Return a writable copy for numerical filtering."""
        return np.asarray(self.covariance, dtype=float).copy()

    def __getitem__(self, key: str) -> object:
        fixed = {
            "report_id": self.report_id,
            "source_id": self.source_id,
            "acquisition_time_s": self.acquisition_time_s,
            "sensor_type": self.sensor_type,
        }
        if key in fixed:
            return fixed[key]
        return self.metadata[key]

    def __iter__(self):
        yield from ("report_id", "source_id", "acquisition_time_s", "sensor_type")
        yield from self.metadata

    def __len__(self) -> int:
        return 4 + len(self.metadata)


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    """Immutable state estimate exported by the operational tracker."""

    track_id: int
    state_time_s: float
    state: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    confidence: float
    lifecycle: TrackLifecycle
    classification_probabilities: tuple[float, ...] | None = None
    classification_entropy_nats: float | None = None
    effective_classification_evidence: float = 0.0
    source_ids: tuple[int | str, ...] = ()
    report_lineage: ReportLineage = ()
    age_s: float = 0.0
    last_measurement_time_s: float | None = None
    lifetime_s: float = 0.0
    engageable: bool = False
    reference_frame: FrameReference | None = None
    suspect_deception: bool = False
    emitter_hypothesis_id: str | None = None

    @property
    def classification(self) -> str:
        if self.classification_probabilities is None:
            return "unknown"
        return CLASSIFICATION_LABELS[int(np.argmax(self.classification_probabilities))]

    def __post_init__(self) -> None:
        state = _tuple(self.state, expected=6)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "covariance", _covariance(self.covariance, 6))
        object.__setattr__(self, "confidence", min(1.0, max(0.0, float(self.confidence))))
        last_measurement_time_s = self.last_measurement_time_s
        if last_measurement_time_s is None:
            last_measurement_time_s = float(self.state_time_s) - max(0.0, float(self.age_s))
        object.__setattr__(self, "last_measurement_time_s", float(last_measurement_time_s))
        object.__setattr__(
            self,
            "age_s",
            max(0.0, float(self.state_time_s) - float(last_measurement_time_s)),
        )
        object.__setattr__(self, "lifetime_s", max(0.0, float(self.lifetime_s)))
        object.__setattr__(
            self,
            "effective_classification_evidence",
            max(0.0, float(self.effective_classification_evidence)),
        )
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        object.__setattr__(self, "report_lineage", canonical_report_lineage(self.report_lineage))
        if self.classification_probabilities is not None:
            probabilities = _tuple(self.classification_probabilities)
            if any(value < 0.0 for value in probabilities) or not np.isclose(
                sum(probabilities), 1.0
            ):
                raise ValueError("Classification probabilities must be nonnegative and sum to 1.")
            object.__setattr__(self, "classification_probabilities", probabilities)
            entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
            object.__setattr__(self, "classification_entropy_nats", float(entropy))
        elif self.classification_entropy_nats is not None:
            object.__setattr__(self, "classification_entropy_nats", None)

    def _with_rigid_frame_transform(
        self,
        state,
        covariance,
        reference_frame: FrameReference,
    ) -> TrackSnapshot:
        """Copy this validated snapshot after a trusted rigid frame transform.

        An orthogonal congruence transform preserves covariance symmetry and positive
        semidefiniteness. Re-running the complete public-constructor validation for
        every receiver therefore proves invariants already established here and in
        the frame-transform helper. This private path still converts arrays to tuples
        at the immutable boundary; all non-geometric fields remain unchanged.
        """
        transformed = copy(self)
        object.__setattr__(
            transformed,
            "state",
            tuple(np.asarray(state, dtype=float).tolist()),
        )
        object.__setattr__(
            transformed,
            "covariance",
            tuple(map(tuple, np.asarray(covariance, dtype=float).tolist())),
        )
        object.__setattr__(transformed, "reference_frame", reference_frame)
        return transformed

    def propagated(self, time_s: float, maneuver_accel_std_mps2: float = 10.0) -> TrackSnapshot:
        """Coast a CV snapshot with bounded unknown-acceleration uncertainty."""
        dt = max(0.0, float(time_s) - self.state_time_s)
        state = np.asarray(self.state, dtype=float)
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        propagated_state = transition @ state
        propagated_covariance = transition @ np.asarray(self.covariance) @ transition.T
        acceleration_variance = float(maneuver_accel_std_mps2) ** 2
        if acceleration_variance > 0.0 and dt > 0.0:
            noise = np.zeros((6, 6), dtype=float)
            noise[:3, :3] = np.eye(3) * (0.25 * acceleration_variance * dt**4)
            noise[:3, 3:] = np.eye(3) * (0.5 * acceleration_variance * dt**3)
            noise[3:, :3] = noise[:3, 3:]
            noise[3:, 3:] = np.eye(3) * (acceleration_variance * dt**2)
            propagated_covariance += noise
        return TrackSnapshot(
            track_id=self.track_id,
            state_time_s=float(time_s),
            state=tuple(propagated_state),
            covariance=tuple(tuple(row) for row in propagated_covariance),
            confidence=self.confidence,
            lifecycle=TrackLifecycle.COASTING,
            classification_probabilities=self.classification_probabilities,
            classification_entropy_nats=self.classification_entropy_nats,
            effective_classification_evidence=self.effective_classification_evidence,
            source_ids=self.source_ids,
            report_lineage=self.report_lineage,
            last_measurement_time_s=self.last_measurement_time_s,
            lifetime_s=self.lifetime_s + dt,
            engageable=self.engageable,
            reference_frame=self.reference_frame,
            suspect_deception=self.suspect_deception,
            emitter_hypothesis_id=self.emitter_hypothesis_id,
        )


@dataclass(frozen=True, slots=True)
class WeaponTrack:
    """Weapon-safe estimate captured at launch or received during midcourse."""

    snapshot: TrackSnapshot
    launch_time_s: float
    update_sequence: int = 0

    def estimate_at(self, time_s: float) -> TrackSnapshot:
        return self.snapshot.propagated(time_s)

    def with_update(self, snapshot: TrackSnapshot) -> WeaponTrack:
        if snapshot.track_id != self.snapshot.track_id:
            raise ValueError("A weapon-track update cannot change contact identity.")
        if snapshot.state_time_s < self.snapshot.state_time_s:
            raise ValueError("A weapon-track update cannot move backward in time.")
        return WeaponTrack(snapshot, self.launch_time_s, self.update_sequence + 1)
