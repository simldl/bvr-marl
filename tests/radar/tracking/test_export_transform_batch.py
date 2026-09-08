"""Equivalence guard for the batched export-frame transform used by
build_track_snapshots. The vectorized form must equal the per-track scalar
transform_state_between_refs to machine epsilon, including identity (from==to)
rows that the scalar path short-circuits.
"""

import numpy as np

from bvr_marl_core.radar.tracking.helpers.recenter_logic import (
    transform_state_between_refs,
    transform_states_between_refs_batch,
)
from bvr_marl_core.simulator.core.helpers import Position

TOL = 1e-9


def _max_rel_err(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(np.abs(b), 1.0)
    return float(np.max(np.abs(a - b) / denom))


def _random_spd(rng) -> np.ndarray:
    a = rng.normal(size=(6, 6))
    return a @ a.T + 6.0 * np.eye(6)


def test_batch_matches_per_track_scalar():
    rng = np.random.default_rng(2026)
    to_ref = Position(12.0, 34.0, 8000.0)

    worst = 0.0
    for _ in range(200):
        N = int(rng.integers(1, 8))
        states, covs, from_refs = [], [], []
        for j in range(N):
            states.append(rng.normal(scale=500.0, size=6))
            covs.append(_random_spd(rng))
            # ~1 in 5 tracks shares the export ref exactly (identity early-out path).
            if rng.random() < 0.2:
                from_refs.append(Position(to_ref.lat, to_ref.lon, to_ref.alt))
            else:
                from_refs.append(
                    Position(
                        to_ref.lat + rng.uniform(-0.4, 0.4),
                        to_ref.lon + rng.uniform(-0.4, 0.4),
                        to_ref.alt + rng.uniform(-2000, 2000),
                    )
                )

        X_to, P_to = transform_states_between_refs_batch(states, covs, from_refs, to_ref)
        for j in range(N):
            xs, Ps = transform_state_between_refs(states[j], covs[j], from_refs[j], to_ref)
            worst = max(worst, _max_rel_err(X_to[j], xs), _max_rel_err(P_to[j], Ps))

    assert worst < TOL, f"batched export transform rel err {worst:.2e} exceeds {TOL:.0e}"


def test_identity_rows_are_exact():
    """A row whose from_ref equals to_ref must pass state/cov through unchanged."""
    rng = np.random.default_rng(7)
    to_ref = Position(-5.0, 100.0, 3000.0)
    state = rng.normal(scale=500.0, size=6)
    cov = _random_spd(rng)

    X_to, P_to = transform_states_between_refs_batch(
        [state], [cov], [Position(to_ref.lat, to_ref.lon, to_ref.alt)], to_ref
    )
    assert np.array_equal(X_to[0], state), "identity row must reproduce state exactly"
    assert np.array_equal(P_to[0], cov), "identity row must reproduce covariance exactly"


def test_shared_source_frame_batch_matches_scalar():
    """The common-picture fast path preserves the authoritative scalar transform."""
    rng = np.random.default_rng(11)
    from_ref = Position(48.1, 11.5, 9000.0)
    to_ref = Position(48.4, 12.0, 7000.0)
    states = [rng.normal(scale=20_000.0, size=6) for _ in range(32)]
    covariances = [_random_spd(rng) for _ in states]

    states_out, covariances_out = transform_states_between_refs_batch(
        states,
        covariances,
        [from_ref] * len(states),
        to_ref,
    )

    for index, (state, covariance) in enumerate(zip(states, covariances, strict=True)):
        expected_state, expected_covariance = transform_state_between_refs(
            state, covariance, from_ref, to_ref
        )
        assert _max_rel_err(states_out[index], expected_state) < TOL
        assert _max_rel_err(covariances_out[index], expected_covariance) < TOL
