import numpy as np
import pytest

from bvr_marl_core.radar.core.utils import enu_to_geodetic, geodetic_to_enu, to_cart
from bvr_marl_core.radar.tracking.filter.constant_velocity_filter import ConstantVelocityKFFilter

# Helpers under test
from bvr_marl_core.radar.tracking.helpers.enu_utils import enu_rotation
from bvr_marl_core.radar.tracking.helpers.measurement_builder import cluster_to_track_measurement
from bvr_marl_core.radar.tracking.helpers.recenter_logic import (
    get_state_in_ref,
    maybe_recenter_reference,
    transform_state_between_refs,
)
from bvr_marl_core.simulator.core.helpers import Position


# --- Reference fixtures (Berlin-ish, >1 km apart) ---
@pytest.fixture
def ref_A():
    return Position(lat=52.0, lon=13.0, alt=50.0)


@pytest.fixture
def ref_B():
    # ~1.3–2.0 km away depending on latitude, enough to trigger recentering
    return Position(lat=52.010, lon=13.020, alt=80.0)


# --- Core rotation properties ---
def test_enu_rotation_properties(ref_A, ref_B):
    R = enu_rotation(ref_A.lat, ref_A.lon, ref_B.lat, ref_B.lon)
    eye = R.T @ R
    assert np.allclose(eye, np.eye(3), atol=1e-9)
    detR = np.linalg.det(R)
    assert np.isclose(detR, 1.0, atol=1e-9)


# --- Identity transform A->A ---
def test_identity_transform(ref_A):
    x = np.array([100.0, -200.0, 30.0, 50.0, 5.0, -2.0])
    P = np.diag([30, 30, 30, 100, 100, 100]).astype(float)

    x2, P2 = transform_state_between_refs(x, P, ref_A, ref_A)
    assert np.allclose(x2, x, atol=1e-9)
    assert np.allclose(P2, P, atol=1e-9)


# --- Round-trip A->B->A ---
def test_roundtrip_transform(ref_A, ref_B):
    x = np.array([250.0, 400.0, -120.0, 10.0, -20.0, 5.0])
    P = np.diag([100, 50, 80, 400, 400, 400]).astype(float)

    xB, PB = transform_state_between_refs(x, P, ref_A, ref_B)
    xA, PA = transform_state_between_refs(xB, PB, ref_B, ref_A)

    assert np.allclose(xA, x, atol=1e-6)
    assert np.allclose(PA, P, atol=1e-6)


# --- Δ direction: old origin in new ENU ---
def test_delta_direction_matches_geodetic_to_enu(ref_A, ref_B):
    # State at the OLD origin (pos=0) must map to Δ = ENU_B(A)
    x = np.zeros(6)
    xB, _ = transform_state_between_refs(x, None, ref_A, ref_B)
    delta_expected = geodetic_to_enu(
        ref_A.lat, ref_A.lon, ref_A.alt, ref_B.lat, ref_B.lon, ref_B.alt
    )
    assert np.allclose(xB[:3], delta_expected, atol=1e-6)
    assert np.allclose(xB[3:], 0.0, atol=1e-12)


# --- Velocity rotates only, position rotates+translates ---
def test_velocity_rotates_only(ref_A, ref_B):
    vA = np.array([120.0, -30.0, 15.0])
    xA = np.concatenate([np.zeros(3), vA])
    xB, _ = transform_state_between_refs(xA, None, ref_A, ref_B)

    # Build expected from R and Δ explicitly
    R = enu_rotation(ref_A.lat, ref_A.lon, ref_B.lat, ref_B.lon)
    delta = geodetic_to_enu(ref_A.lat, ref_A.lon, ref_A.alt, ref_B.lat, ref_B.lon, ref_B.alt)

    assert np.allclose(xB[:3], delta, atol=1e-6)  # pos = R*0 + Δ
    assert np.allclose(xB[3:6], R @ vA, atol=1e-9)  # vel = R*v


# --- Covariance block rotation: S = blkdiag(R,R) ---
def test_covariance_block_rotation(ref_A, ref_B):
    # Build a symmetric PSD covariance with cross-terms
    rng = np.random.default_rng(0)
    M = rng.standard_normal((6, 6))
    P = M @ M.T + np.eye(6) * 1e-3

    x = np.zeros(6)
    xB, PB = transform_state_between_refs(x, P, ref_A, ref_B)

    R = enu_rotation(ref_A.lat, ref_A.lon, ref_B.lat, ref_B.lon)
    S = np.zeros((6, 6))
    S[:3, :3] = R
    S[3:, 3:] = R
    P_expected = S @ P @ S.T

    assert np.allclose(PB, P_expected, atol=1e-8)


# --- Recenter equivalence: maybe_recenter_reference == generic transform ---
def test_maybe_recenter_matches_generic(ref_A, ref_B):
    # Start the KF at some non-zero state in ref_A
    measA = np.array([300.0, -150.0, 40.0])
    kf = ConstantVelocityKFFilter(measA, dt=1.0, process_noise_std=25.0, measurement_noise_var=1.0)

    tracks = {42: kf}
    track_refs = {42: Position(ref_A.lat, ref_A.lon, ref_A.alt)}

    # Expected transform via generic helper
    x_before = kf.get_state().copy()
    P_before = kf.get_covariance().copy()
    x_expected, P_expected = transform_state_between_refs(x_before, P_before, ref_A, ref_B)

    # Force a recenter by calling maybe_recenter_reference with ref_B
    did = maybe_recenter_reference(42, track_refs, tracks, ref_B)
    assert did is True, "Drift should exceed threshold so recenter occurs"

    x_after = tracks[42].get_state()
    P_after = tracks[42].get_covariance()

    # They must match bit-for-bit (up to numerical tolerance)
    assert np.allclose(x_after, x_expected, atol=1e-9)
    assert np.allclose(P_after, P_expected, atol=1e-9)

    # And the stored reference must be updated to B
    assert np.isclose(track_refs[42].lat, ref_B.lat)
    assert np.isclose(track_refs[42].lon, ref_B.lon)
    assert np.isclose(track_refs[42].alt, ref_B.alt)


# --- Measurement fallback: rotate+translate equals geodetic path ---
def test_cluster_fallback_matches_manual_rotate_translate(ref_B):
    src = Position(lat=52.005, lon=13.015, alt=60.0)
    az_deg, el_deg, d_m = 45.0, 5.0, 20000.0
    cluster = {"az": az_deg, "el": el_deg, "d": d_m, "source_pos": src}

    # Function under test
    p_ref_fallback = cluster_to_track_measurement(cluster, ref_B)

    # Manual (the same math the function is meant to do)
    from bvr_marl_core.radar.tracking.helpers.enu_utils import enu_rotation

    v_local = np.array(to_cart(az_deg, el_deg, d_m), dtype=float)
    p_src_in_ref = geodetic_to_enu(src.lat, src.lon, src.alt, ref_B.lat, ref_B.lon, ref_B.alt)
    R = enu_rotation(src.lat, src.lon, ref_B.lat, ref_B.lon)
    p_manual = p_src_in_ref + R @ v_local

    assert np.allclose(p_ref_fallback, p_manual, atol=1e-9)


def test_cluster_fallback_vs_geodetic_short_range(ref_B):
    src = Position(lat=52.005, lon=13.015, alt=60.0)
    az_deg, el_deg, d_m = 45.0, 5.0, 3000.0  # 3 km keeps linearization tight
    cluster = {"az": az_deg, "el": el_deg, "d": d_m, "source_pos": src}

    p_ref_fallback = cluster_to_track_measurement(cluster, ref_B)

    lat, lon, alt = enu_to_geodetic(
        np.array(to_cart(az_deg, el_deg, d_m)), src.lat, src.lon, src.alt
    )
    p_ref_geodetic = geodetic_to_enu(lat, lon, alt, ref_B.lat, ref_B.lon, ref_B.alt)

    # sub-meter to a few meters depending on latitude/elevation; be modest:
    assert np.allclose(p_ref_fallback, p_ref_geodetic, atol=5.0)
