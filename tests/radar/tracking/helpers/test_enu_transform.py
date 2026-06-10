"""
Unit tests locking the ENU-to-ENU state-transform convention in recenter_logic.py.

Convention (from transform_state_between_refs docstring):
    p_to = R * p_from + t_from_origin_in_to
    where t_from_origin_in_to = geodetic_to_enu(from_ref, ref=to_ref)
                               = position of from_ref expressed in ENU(to_ref)
"""

import numpy as np
import pytest

from bvr_marl_core.radar.core.utils import geodetic_to_enu
from bvr_marl_core.radar.tracking.helpers.recenter_logic import transform_state_between_refs
from bvr_marl_core.simulator.core.helpers import Position

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pos(lat, lon, alt=0.0):
    p = Position(lat, lon, alt)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTransformStateBetweenRefs:
    def test_sign_of_translation(self):
        """
        to_ref is ~1000 m East of from_ref.

        t_from_origin_in_to must point WEST (negative East) because from_ref is
        West of to_ref when viewed from to_ref's origin.
        """
        # 1° lon ≈ 111320 m at equator → 1000 m ≈ 0.008983°
        dlon = 1000.0 / 111_320.0
        from_ref = make_pos(0.0, 0.0)
        to_ref = make_pos(0.0, dlon)  # ~1000 m East of from_ref

        t_from_origin_in_to = geodetic_to_enu(
            from_ref.lat,
            from_ref.lon,
            from_ref.alt,
            to_ref.lat,
            to_ref.lon,
            to_ref.alt,
        )
        # from_ref is West of to_ref → East component should be negative
        assert t_from_origin_in_to[0] < -900.0, (
            f"Expected t[East] < -900 m; got {t_from_origin_in_to[0]:.1f} m"
        )
        assert abs(t_from_origin_in_to[1]) < 1.0, (
            f"Expected t[North] ≈ 0; got {t_from_origin_in_to[1]:.4f} m"
        )

    def test_round_trip_position(self):
        """
        A point expressed in ENU(from_ref) transforms to ENU(to_ref) and back.
        The round-trip must be the identity.
        """
        from_ref = make_pos(48.0, 11.0)
        to_ref = make_pos(48.01, 11.02)  # ~1.5 km NE

        # Arbitrary 6-D state in ENU(from_ref)
        state6 = np.array(
            [
                500.0,
                -200.0,
                300.0,  # position (m)
                50.0,
                -30.0,
                10.0,
            ]
        )  # velocity (m/s)

        # Forward: from_ref → to_ref
        state_to, _ = transform_state_between_refs(state6, None, from_ref, to_ref)

        # Inverse: to_ref → from_ref
        state_back, _ = transform_state_between_refs(state_to, None, to_ref, from_ref)

        np.testing.assert_allclose(
            state_back, state6, atol=1e-6, err_msg="Round-trip transform must be identity"
        )

    def test_stationary_target_at_to_ref(self):
        """
        A target sitting exactly at to_ref should appear at the origin (0,0,0)
        in ENU(to_ref), regardless of the from_ref.
        """
        from_ref = make_pos(0.0, 0.0)
        to_ref = make_pos(0.0, 0.009)  # ~1000 m East

        # In ENU(from_ref), to_ref is ~1000 m East
        p_to_ref_in_from = geodetic_to_enu(
            to_ref.lat,
            to_ref.lon,
            to_ref.alt,
            from_ref.lat,
            from_ref.lon,
            from_ref.alt,
        )
        state6 = np.concatenate([p_to_ref_in_from, [0.0, 0.0, 0.0]])

        state_out, _ = transform_state_between_refs(state6, None, from_ref, to_ref)

        np.testing.assert_allclose(
            state_out[:3],
            [0.0, 0.0, 0.0],
            atol=0.1,
            err_msg="Target at to_ref must map to origin in ENU(to_ref)",
        )

    def test_covariance_transforms(self):
        """
        A spherical position covariance remains spherical after transformation
        (rotation preserves isotropy).  Off-diagonal terms must remain zero.
        """
        from_ref = make_pos(45.0, 10.0)
        to_ref = make_pos(45.1, 10.1)

        state6 = np.array([100.0, 200.0, 50.0, 10.0, -5.0, 2.0])
        sigma_pos = 50.0  # m, isotropic
        sigma_vel = 5.0  # m/s, isotropic
        P6 = np.diag([sigma_pos**2] * 3 + [sigma_vel**2] * 3)

        _, P_to = transform_state_between_refs(state6, P6, from_ref, to_ref)

        assert P_to is not None
        assert P_to.shape == (6, 6)
        # Diagonal should be preserved (rotation preserves trace)
        np.testing.assert_allclose(
            np.diag(P_to),
            np.diag(P6),
            rtol=1e-6,
            err_msg="Isotropic covariance diagonal must be preserved by rotation",
        )
        # Must remain positive definite
        eigvals = np.linalg.eigvalsh(P_to)
        assert np.all(eigvals > 0), f"Transformed covariance not PD; min eigval={eigvals.min():.2e}"

    def test_zero_separation_is_identity(self):
        """
        When from_ref == to_ref, the transform is the identity (R=I, t=0).
        """
        ref = make_pos(51.5, -0.1)
        state6 = np.array([1000.0, -500.0, 200.0, 20.0, 10.0, -5.0])
        P6 = np.eye(6) * 100.0

        state_out, P_out = transform_state_between_refs(state6, P6, ref, ref)

        np.testing.assert_allclose(state_out, state6, atol=1e-8)
        np.testing.assert_allclose(P_out, P6, atol=1e-8)
