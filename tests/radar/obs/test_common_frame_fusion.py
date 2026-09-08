import math

import numpy as np
import pytest

from bvr_marl_core.domain.information import FrameReference, SensorReport, SensorType
from bvr_marl_core.radar.core.utils import geodetic_to_enu
from bvr_marl_core.radar.obs.common_frame import (
    CartesianReport,
    _connected_report_groups,
    _fuse_group,
    _pair_gate_distance,
    _source_cartesian_terms,
    cluster_reports_common_frame,
    report_in_receiver_enu,
    spherical_to_cartesian_jacobian,
)
from bvr_marl_core.simulator.core.helpers import Position


def _spherical(enu):
    east, north, up = enu
    horizontal = math.hypot(east, north)
    return (
        math.degrees(math.atan2(east, north)),
        math.degrees(math.atan2(up, horizontal)),
        math.hypot(horizontal, up),
    )


def _report(source: Position, target: Position, source_id: int, report_id: int = 1):
    measurement = _spherical(
        geodetic_to_enu(
            target.lat,
            target.lon,
            target.alt,
            source.lat,
            source.lon,
            source.alt,
        )
    )
    return SensorReport(
        report_id=report_id,
        source_id=source_id,
        acquisition_time_s=float(source_id),
        measurement=measurement,
        covariance=np.diag([0.01**2, 0.01**2, 10.0**2]),
        frame=FrameReference(source.lat, source.lon, source.alt),
        sensor_type=SensorType.RADAR,
        classification_probabilities=(0.8, 0.05, 0.05, 0.02, 0.08),
    )


def test_spherical_covariance_jacobian_matches_finite_difference():
    measurement = np.array([31.0, -7.0, 42_000.0])
    analytical = spherical_to_cartesian_jacobian(*measurement)

    def cartesian(value):
        azimuth, elevation, radius = np.radians(value[0]), np.radians(value[1]), value[2]
        return np.array(
            [
                radius * np.cos(elevation) * np.sin(azimuth),
                radius * np.cos(elevation) * np.cos(azimuth),
                radius * np.sin(elevation),
            ]
        )

    numerical = np.empty((3, 3))
    steps = (1e-5, 1e-5, 1e-2)
    for column, step in enumerate(steps):
        delta = np.zeros(3)
        delta[column] = step
        numerical[:, column] = (cartesian(measurement + delta) - cartesian(measurement - delta)) / (
            2.0 * step
        )

    np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-6)


def test_source_cartesian_terms_are_reused_across_receivers():
    _source_cartesian_terms.cache_clear()
    source = Position(0.0, 0.0, 9_000.0)
    target = Position(0.2, 0.3, 10_000.0)
    report = _report(source, target, source_id=4)

    first = report_in_receiver_enu(report, Position(0.0, 0.0, 9_000.0))
    cache_after_first = _source_cartesian_terms.cache_info()
    second = report_in_receiver_enu(report, Position(0.1, -0.1, 8_000.0))
    cache_after_second = _source_cartesian_terms.cache_info()

    assert cache_after_second.hits == cache_after_first.hits + 1
    assert np.all(np.isfinite(first.position_enu_m))
    assert np.all(np.isfinite(second.covariance_enu_m2))


@pytest.mark.parametrize("sensor_count", [1, 2, 4, 8])
def test_one_target_from_separated_sensors_produces_one_common_frame_cluster(sensor_count):
    receiver = Position(48.0, 11.0, 5_000.0)
    target = Position(48.35, 11.15, 7_000.0)
    sources = [
        Position(48.0 + 0.02 * index, 11.0 - 0.025 * index, 5_000.0 + 100.0 * index)
        for index in range(sensor_count)
    ]
    reports = [_report(source, target, index + 1) for index, source in enumerate(sources)]

    clusters = cluster_reports_common_frame(reports, receiver)

    assert len(clusters) == 1
    assert clusters[0]["n_obs"] == sensor_count
    assert len(clusters[0]["report_lineage"]) == sensor_count
    assert np.argmax(clusters[0]["classification_probabilities"]) == 0
    expected = geodetic_to_enu(
        target.lat,
        target.lon,
        target.alt,
        receiver.lat,
        receiver.lon,
        receiver.alt,
    )
    actual = geodetic_to_enu(
        clusters[0]["lat"],
        clusters[0]["lon"],
        clusters[0]["alt"],
        receiver.lat,
        receiver.lon,
        receiver.alt,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_same_local_report_id_from_two_sensors_remains_source_qualified():
    receiver = Position(48.0, 11.0, 5_000.0)
    target = Position(48.35, 11.15, 7_000.0)
    reports = [
        _report(Position(48.0, 11.0, 5_000.0), target, 10, 1),
        _report(Position(48.01, 10.99, 5_100.0), target, 20, 1),
    ]

    clusters = cluster_reports_common_frame(reports, receiver)

    assert len(clusters) == 1
    assert clusters[0]["report_lineage"] == ((10, 1), (20, 1))
    assert clusters[0]["n_obs"] == 2


def test_relayed_copy_of_same_report_is_not_independent_evidence():
    receiver = Position(48.0, 11.0, 5_000.0)
    report = _report(receiver, Position(48.35, 11.15, 7_000.0), 10, 1)

    clusters = cluster_reports_common_frame([report, report, report], receiver)

    assert len(clusters) == 1
    assert clusters[0]["report_lineage"] == ((10, 1),)
    assert clusters[0]["n_obs"] == 1


def test_repeated_same_source_classification_keeps_one_effective_opinion():
    receiver = Position(48.0, 11.0, 5_000.0)
    target = Position(48.35, 11.15, 7_000.0)
    reports = [_report(receiver, target, 10, report_id) for report_id in range(1, 11)]

    one = cluster_reports_common_frame(reports[:1], receiver)[0]
    repeated = cluster_reports_common_frame(reports, receiver)[0]

    assert repeated["effective_classification_evidence"] == 1.0
    assert repeated["classification_entropy_nats"] == pytest.approx(
        one["classification_entropy_nats"]
    )
    assert repeated["classification_probabilities"] == pytest.approx(
        one["classification_probabilities"]
    )


def test_independent_sensors_reduce_entropy_more_than_retransmissions():
    receiver = Position(48.0, 11.0, 5_000.0)
    target = Position(48.35, 11.15, 7_000.0)
    first = _report(receiver, target, 10, 1)
    second = _report(Position(48.01, 10.99, 5_100.0), target, 20, 1)

    relayed = cluster_reports_common_frame([first, first], receiver)[0]
    independent = cluster_reports_common_frame([first, second], receiver)[0]

    assert relayed["effective_classification_evidence"] == 1.0
    assert independent["effective_classification_evidence"] == 2.0
    assert independent["classification_entropy_nats"] < relayed["classification_entropy_nats"]


def test_transformed_covariance_is_symmetric_psd_and_keeps_correlations():
    receiver = Position(48.0, 11.0, 5_000.0)
    report = _report(Position(47.9, 10.8, 3_000.0), Position(48.2, 11.3, 8_000.0), 2)

    converted = report_in_receiver_enu(report, receiver)

    covariance = converted.covariance_enu_m2
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-10)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-8
    assert np.count_nonzero(np.abs(covariance - np.diag(np.diag(covariance))) > 1e-6) > 0


def test_separated_targets_are_not_merged():
    receiver = Position(48.0, 11.0, 5_000.0)
    source = Position(48.0, 11.0, 5_000.0)
    first = Position(48.2, 11.0, 6_000.0)
    second = Position(48.2009, 11.0, 6_000.0)

    clusters = cluster_reports_common_frame(
        [_report(source, first, 1, 1), _report(source, second, 1, 2)], receiver
    )

    assert len(clusters) == 2


def test_impossible_report_pair_is_rejected_before_matrix_solve(monkeypatch):
    source = Position(48.0, 11.0, 5_000.0)
    target = Position(48.1, 11.0, 5_000.0)
    report = _report(source, target, 1)
    covariance = np.eye(3) * 10_000.0
    cartesian = [
        CartesianReport(report, np.zeros(3), covariance),
        CartesianReport(_report(source, target, 1, 2), np.full(3, 800.0), covariance),
    ]

    def unexpected_solve(*_args, **_kwargs):
        raise AssertionError("an analytically impossible pair must not be factorised")

    monkeypatch.setattr(np.linalg, "solve", unexpected_solve)
    monkeypatch.setattr(np.linalg, "pinv", unexpected_solve)

    groups = _connected_report_groups(cartesian)

    assert len(groups) == 2


@pytest.mark.parametrize("singular", [False, True])
def test_batched_information_fusion_matches_individual_pseudoinverses(singular):
    rng = np.random.default_rng(20260721)
    receiver = Position(48.0, 11.0, 5_000.0)
    target = Position(48.2, 11.1, 6_000.0)
    cartesian = []
    for index in range(6):
        matrix = rng.normal(size=(3, 3))
        covariance = matrix @ matrix.T + np.eye(3)
        if singular:
            covariance[2] = 0.0
            covariance[:, 2] = 0.0
        cartesian.append(
            CartesianReport(
                _report(receiver, target, index + 1),
                rng.normal(scale=10_000.0, size=3),
                covariance,
            )
        )

    inverse_covariances = [np.linalg.pinv(item.covariance_enu_m2) for item in cartesian]
    expected_covariance = np.linalg.pinv(sum(inverse_covariances, np.zeros((3, 3))))
    expected_position = expected_covariance @ sum(
        (
            inverse @ item.position_enu_m
            for inverse, item in zip(inverse_covariances, cartesian, strict=True)
        ),
        np.zeros(3),
    )

    fused = _fuse_group(cartesian, receiver)
    actual_position = geodetic_to_enu(
        fused["lat"],
        fused["lon"],
        fused["alt"],
        receiver.lat,
        receiver.lon,
        receiver.alt,
    )
    np.testing.assert_allclose(fused["covariance_cartesian"], expected_covariance, rtol=1e-12)
    np.testing.assert_allclose(actual_position, expected_position, rtol=1e-10, atol=1e-6)


def test_delayed_report_uncertainty_grows_with_age():
    receiver = Position(48.0, 11.0, 5_000.0)
    report = _report(receiver, Position(48.2, 11.1, 6_000.0), 1)

    fresh = report_in_receiver_enu(report, receiver, current_time_s=report.acquisition_time_s)
    delayed = report_in_receiver_enu(
        report,
        receiver,
        current_time_s=report.acquisition_time_s + 5.0,
    )

    assert np.all(np.diag(delayed.covariance_enu_m2) > np.diag(fresh.covariance_enu_m2))


def test_reports_older_than_retained_policy_are_rejected():
    receiver = Position(48.0, 11.0, 5_000.0)
    report = _report(receiver, Position(48.2, 11.1, 6_000.0), 1)

    clusters = cluster_reports_common_frame(
        [report],
        receiver,
        current_time_s=report.acquisition_time_s + 31.0,
        max_report_age_s=30.0,
    )

    assert clusters == []


def _cartesian(position_enu, sigma_m, source_id, report_id=1):
    """A report already in the receiver frame, with isotropic uncertainty."""
    source = Position(0.0, 0.0, 9_000.0)
    return CartesianReport(
        report=_report(source, source, source_id=source_id, report_id=report_id),
        position_enu_m=np.asarray(position_enu, dtype=float),
        covariance_enu_m2=np.eye(3) * float(sigma_m) ** 2,
    )


def test_grouping_does_not_chain_incompatible_reports_through_an_intermediate():
    """Single-linkage would fuse the two ends into one contact via the middle report.

    A and C are far enough apart to fail the gate outright; B sits between them and
    gates against both. Complete linkage must keep A and C in separate groups, because
    a chained group is a track that covers several aircraft and identifies none.
    """
    sigma = 1_000.0
    a = _cartesian((0.0, 0.0, 0.0), sigma, source_id=1)
    b = _cartesian((3_500.0, 0.0, 0.0), sigma, source_id=2)
    c = _cartesian((7_000.0, 0.0, 0.0), sigma, source_id=3)

    assert _pair_gate_distance(a, b) is not None
    assert _pair_gate_distance(b, c) is not None
    assert _pair_gate_distance(a, c) is None, "precondition: the ends must not gate"

    groups = _connected_report_groups([a, b, c])

    positions = [{float(item.position_enu_m[0]) for item in group} for group in groups]
    # B may join either end -- it genuinely gates against both -- but the two ends
    # must never end up describing one contact.
    assert not any({0.0, 7_000.0} <= group for group in positions)
    assert len(groups) > 1


def test_relayed_duplicates_coalesce_without_passing_the_gate():
    """The same physical observation delivered twice is one contact, however far apart."""
    original = _cartesian((0.0, 0.0, 0.0), 10.0, source_id=1, report_id=77)
    relayed = _cartesian((50_000.0, 0.0, 0.0), 10.0, source_id=1, report_id=77)

    assert _pair_gate_distance(original, relayed) is None
    assert len(_connected_report_groups([original, relayed])) == 1


def test_grouping_is_independent_of_report_order():
    reports = [
        _cartesian((0.0, 0.0, 0.0), 800.0, source_id=1),
        _cartesian((2_400.0, 0.0, 0.0), 800.0, source_id=2),
        _cartesian((12_000.0, 0.0, 0.0), 800.0, source_id=3),
        _cartesian((14_000.0, 0.0, 0.0), 800.0, source_id=4),
    ]
    expected = sorted(
        sorted(float(item.position_enu_m[0]) for item in group)
        for group in _connected_report_groups(reports)
    )
    for rotation in range(1, len(reports)):
        rotated = reports[rotation:] + reports[:rotation]
        actual = sorted(
            sorted(float(item.position_enu_m[0]) for item in group)
            for group in _connected_report_groups(rotated)
        )
        assert actual == expected
