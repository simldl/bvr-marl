"""Common-frame conversion and covariance-aware report grouping."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from bvr_marl_core.domain.information import SensorReport
from bvr_marl_core.radar.core.utils import enu_to_geodetic
from bvr_marl_core.radar.tracking.helpers.recenter_logic import _frame_rotation_translation
from bvr_marl_core.simulator.core.helpers import Position

_GATE_3D_999 = 16.266


@dataclass(frozen=True, slots=True)
class CartesianReport:
    """One immutable report expressed in a receiver's ENU frame."""

    report: SensorReport
    position_enu_m: np.ndarray
    covariance_enu_m2: np.ndarray


def spherical_to_cartesian_jacobian(az_deg: float, el_deg: float, range_m: float) -> np.ndarray:
    """Jacobian for ``(az_deg, el_deg, range_m) -> ENU metres``.

    The angular columns include the degree-to-radian scale because production
    ``SensorReport`` angular covariance is expressed in degrees squared.
    """
    az = math.radians(float(az_deg))
    el = math.radians(float(el_deg))
    radius = float(range_m)
    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)
    deg = math.pi / 180.0
    return np.array(
        [
            [radius * ce * ca * deg, -radius * se * sa * deg, ce * sa],
            [-radius * ce * sa * deg, -radius * se * ca * deg, ce * ca],
            [0.0, radius * ce * deg, se],
        ],
        dtype=float,
    )


@lru_cache(maxsize=4096)
def _source_cartesian_terms(
    measurement: tuple[float, ...], covariance: tuple[tuple[float, ...], ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Cache receiver-independent work for one immutable sensor report."""
    az_deg, el_deg, range_m = measurement
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    source_vector = np.array(
        [
            range_m * math.cos(el) * math.sin(az),
            range_m * math.cos(el) * math.cos(az),
            range_m * math.sin(el),
        ],
        dtype=float,
    )
    jacobian = spherical_to_cartesian_jacobian(az_deg, el_deg, range_m)
    source_covariance = jacobian @ np.asarray(covariance, dtype=float) @ jacobian.T
    source_vector.setflags(write=False)
    source_covariance.setflags(write=False)
    return source_vector, source_covariance


def report_in_receiver_enu(
    report: SensorReport,
    receiver: Position,
    *,
    current_time_s: float | None = None,
    maneuver_accel_std_mps2: float = 20.0,
) -> CartesianReport:
    """Transform a spherical source report and its covariance to receiver ENU."""
    source_vector, source_covariance = _source_cartesian_terms(
        report.measurement, report.covariance
    )
    source = report.frame
    # rotation = ENU(source)->ENU(receiver); source_origin = source frame origin in
    # receiver ENU. Both are exactly the cached (R, t) of an ENU-to-ENU transform,
    # which repeats across every report sharing this source frame within a tick.
    rotation, source_origin = _frame_rotation_translation(
        source.latitude_deg,
        source.longitude_deg,
        source.altitude_m,
        receiver.lat,
        receiver.lon,
        receiver.alt,
    )
    receiver_covariance = rotation @ source_covariance @ rotation.T
    if current_time_s is not None:
        age_s = max(0.0, float(current_time_s) - float(report.acquisition_time_s))
        # With no reported velocity, retain the acquisition-time position and
        # represent bounded unknown manoeuvre displacement as added uncertainty.
        displacement_std_m = 0.5 * float(maneuver_accel_std_mps2) * age_s**2
        receiver_covariance += np.eye(3) * displacement_std_m**2
    receiver_covariance = 0.5 * (receiver_covariance + receiver_covariance.T)
    return CartesianReport(
        report=report,
        position_enu_m=source_origin + rotation @ source_vector,
        covariance_enu_m2=receiver_covariance,
    )


def _pair_gate_distance(a: CartesianReport, b: CartesianReport) -> float | None:
    """Mahalanobis distance between two reports, or None if they cannot gate."""
    delta = a.position_enu_m - b.position_enu_m
    covariance = a.covariance_enu_m2 + b.covariance_enu_m2
    # For positive-semidefinite covariance,
    # d.T @ inv(C) @ d >= ||d||^2 / trace(C).  Rejecting on this
    # lower bound avoids a matrix factorisation for pairs that
    # cannot possibly pass the statistical gate.
    delta_squared = float(delta @ delta)
    covariance_trace = float(covariance[0, 0] + covariance[1, 1] + covariance[2, 2])
    if delta_squared > _GATE_3D_999 * covariance_trace:
        return None
    try:
        distance = float(delta @ np.linalg.solve(covariance, delta))
    except np.linalg.LinAlgError:
        distance = float(delta @ np.linalg.pinv(covariance) @ delta)
    return distance if distance <= _GATE_3D_999 else None


def _connected_report_groups(reports: Sequence[CartesianReport]) -> list[list[CartesianReport]]:
    """Group statistically compatible reports using coarse bins then a 3-D gate.

    Grouping is **complete-linkage**: a report joins a group only when it gates
    against every member. Single-linkage (transitive union) chains through
    intermediates -- reports A and C that are statistically incompatible still land
    in one group whenever some B gates against both -- which silently fuses distinct
    aircraft into a single track. A formation is exactly the geometry that produces
    such chains, and the fused track then covers several aircraft at once and no
    longer identifies any one of them.
    """
    if not reports:
        return []
    max_sigma = max(
        math.sqrt(max(1.0, float(np.linalg.eigvalsh(item.covariance_enu_m2).max())))
        for item in reports
    )
    cell_size = max(100.0, 4.1 * max_sigma)
    bins: dict[tuple[int, int, int], list[int]] = {}
    for index, item in enumerate(reports):
        cell = tuple(np.floor(item.position_enu_m / cell_size).astype(int))
        bins.setdefault(cell, []).append(index)

    parent = list(range(len(reports)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    # Relayed copies carry the same immutable report id. They are lineage
    # duplicates rather than independent evidence, so they always coalesce and
    # never have to clear the statistical gate.
    by_identity: dict[tuple[object, object], int] = {}
    for index, item in enumerate(reports):
        identity = (item.report.source_id, item.report.report_id)
        first = by_identity.setdefault(identity, index)
        if first != index:
            union(first, index)

    offsets = tuple((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1))
    compatible: dict[tuple[int, int], float] = {}
    for cell, indices in bins.items():
        candidates = {
            other
            for offset in offsets
            for other in bins.get(tuple(cell[axis] + offset[axis] for axis in range(3)), ())
        }
        for left in indices:
            for right in candidates:
                pair = (left, right) if left < right else (right, left)
                if left == right or pair in compatible:
                    continue
                distance = _pair_gate_distance(reports[pair[0]], reports[pair[1]])
                if distance is not None:
                    compatible[pair] = distance

    members: dict[int, list[int]] = {}
    for index in range(len(reports)):
        members.setdefault(find(index), []).append(index)

    def gates(left: int, right: int) -> bool:
        pair = (left, right) if left < right else (right, left)
        return pair in compatible or find(left) == find(right)

    # Merge closest-first, so the result does not depend on report ordering, and
    # accept a merge only when every cross-pair between the two groups gates.
    for left, right in sorted(compatible, key=compatible.__getitem__):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        left_members, right_members = members[left_root], members[right_root]
        if not all(gates(one, other) for one in left_members for other in right_members):
            continue
        union(left_root, right_root)
        merged_root = find(left_root)
        merged_members = left_members + right_members
        for stale in (left_root, right_root):
            if stale != merged_root:
                del members[stale]
        members[merged_root] = merged_members

    grouped: dict[int, list[CartesianReport]] = {}
    for index, report in enumerate(reports):
        grouped.setdefault(find(index), []).append(report)
    return list(grouped.values())


def _fuse_group(group: Sequence[CartesianReport], receiver: Position) -> dict:
    unique = {(item.report.source_id, item.report.report_id): item for item in group}
    independent = list(unique.values())
    covariances = np.stack([item.covariance_enu_m2 for item in independent])
    # NumPy's stacked pseudoinverse performs one batched SVD dispatch. This retains
    # Moore-Penrose behavior for bearing-only/singular reports while avoiding one
    # Python-to-linalg call per independent sensor in the normal multi-report case.
    inverse_covariances = np.linalg.pinv(covariances)
    information = inverse_covariances.sum(axis=0)
    covariance = np.linalg.pinv(information)
    positions = np.stack([item.position_enu_m for item in independent])
    position = covariance @ np.einsum(
        "nij,nj->i",
        inverse_covariances,
        positions,
    )
    lat, lon, alt = enu_to_geodetic(position, receiver.lat, receiver.lon, receiver.alt)
    east, north, up = position
    horizontal = math.hypot(east, north)
    distance = math.hypot(horizontal, up)
    reports = [item.report for item in independent]
    metadata: list[Mapping[str, object]] = [report.metadata for report in reports]
    class_beliefs_by_source: dict[object, list[np.ndarray]] = {}
    for report in reports:
        if report.classification_probabilities is not None:
            class_beliefs_by_source.setdefault(report.source_id, []).append(
                np.asarray(report.classification_probabilities, dtype=float)
            )
    fused_classification = None
    classification_entropy_nats = None
    effective_classification_evidence = 0.0
    if class_beliefs_by_source:
        # Reports from one source share calibration, geometry, and often the
        # same return through several relays. Average their log beliefs into one
        # source opinion; only distinct sensors contribute independent weight.
        source_opinions = [
            np.exp(np.mean(np.log(np.maximum(beliefs, 1e-6)), axis=0))
            for beliefs in class_beliefs_by_source.values()
        ]
        fused = np.prod(source_opinions, axis=0)
        fused_classification = tuple((fused / fused.sum()).tolist())
        effective_classification_evidence = float(len(source_opinions))
        probabilities = np.asarray(fused_classification)
        classification_entropy_nats = float(
            -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
        )
    return {
        "az": math.degrees(math.atan2(east, north)),
        "el": math.degrees(math.atan2(up, horizontal)),
        "d": distance,
        "dop": float(np.mean([float(item.get("dop", 0.0)) for item in metadata])),
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "measurement_position": Position(lat=lat, lon=lon, alt=alt),
        "measurement_ref": receiver.copy(),
        "covariance_cartesian": covariance,
        "n_obs": len(independent),
        "T": None,
        "is_deception": any(bool(item.get("is_deception", False)) for item in metadata),
        "engagement_id": next(
            (
                item.get("engagement_id")
                for item in metadata
                if item.get("engagement_id") is not None
            ),
            None,
        ),
        "jammer_id": next(
            (item.get("jammer_id") for item in metadata if item.get("jammer_id") is not None),
            None,
        ),
        "report_ids": tuple(report.report_id for report in reports),
        "report_lineage": tuple(sorted(unique, key=lambda value: (str(value[0]), value[1]))),
        "source_ids": tuple(sorted({report.source_id for report in reports}, key=str)),
        "acquisition_time_s": max(report.acquisition_time_s for report in reports),
        "sensor_types": tuple(sorted({report.sensor_type.value for report in reports})),
        "classification_probabilities": fused_classification,
        "classification_entropy_nats": classification_entropy_nats,
        "effective_classification_evidence": effective_classification_evidence,
    }


def cluster_reports_common_frame(
    reports: Sequence[SensorReport],
    receiver: Position,
    *,
    current_time_s: float | None = None,
    max_report_age_s: float = 30.0,
) -> list[dict]:
    """Convert, gate, and information-fuse reports in one Cartesian frame."""
    # Collapse relay fan-out before coordinate conversion and O(n²) spatial
    # grouping. A source-qualified report is one physical observation no matter
    # how many network paths delivered it to this receiver.
    accepted_by_lineage = {
        (report.source_id, report.report_id): report
        for report in reports
        if current_time_s is None
        or 0.0 <= float(current_time_s) - float(report.acquisition_time_s) <= max_report_age_s
    }
    accepted = list(accepted_by_lineage.values())
    converted = [
        report_in_receiver_enu(report, receiver, current_time_s=current_time_s)
        for report in accepted
    ]
    return [_fuse_group(group, receiver) for group in _connected_report_groups(converted)]
