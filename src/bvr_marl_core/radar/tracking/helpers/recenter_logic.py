import math
from functools import lru_cache

import numpy as np

from bvr_marl_core.radar.core.utils import geodetic_to_enu
from bvr_marl_core.radar.tracking.helpers.enu_utils import enu_rotation
from bvr_marl_core.simulator.core.helpers import Position

RECENTER_THRESH_M = 1000.0


@lru_cache(maxsize=512)
def _frame_rotation_translation(
    from_lat: float, from_lon: float, from_alt: float, to_lat: float, to_lon: float, to_alt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cached (R, t) for an ENU(from) -> ENU(to) transform. Track references are stable
    between recenters and export refs repeat within a tick, so identical pairs recur
    often. The returned arrays are shared cache entries: callers must NOT mutate them.
    """
    R = enu_rotation(from_lat, from_lon, to_lat, to_lon)
    t = geodetic_to_enu(from_lat, from_lon, from_alt, to_lat, to_lon, to_alt)
    return R, t


def _rotate_covariance_blocks(cov6: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply the blkdiag(R, R) covariance rotation ``S @ cov6 @ S.T``.

    Materializing ``S = blkdiag(R, R)`` and using two 6x6 BLAS matmuls is ~3x
    faster than the per-3x3-block formulation: on these small matrices BLAS call
    overhead dominates, so fewer/larger products win over many tiny ones.
    """
    S = np.zeros((6, 6), dtype=float)
    S[:3, :3] = R
    S[3:, 3:] = R
    return S @ cov6 @ S.T


def maybe_recenter_reference(tid: int, track_refs: dict, tracks: dict, new_ref: Position) -> bool:
    """
    Check if track reference needs recentering and perform transformation.

    When the reference changes, we transform the filter state:
      x_pos' = R * x_pos + Δ
      v'     = R * v
      P'     = S P S^T, with S = blkdiag(R, R)
    where:
      - R maps vectors from old ENU → new ENU
      - Δ is coordinates of OLD reference in NEW ENU

    Args:
        tid: Track ID
        track_refs: Dictionary of track references
        tracks: Dictionary of track filters
        new_ref: New reference position to consider

    Returns:
        True if recentering was performed, False otherwise
    """
    old_ref = track_refs.get(tid)
    if old_ref is None:
        track_refs[tid] = new_ref.copy()
        return False

    if old_ref.lat == new_ref.lat and old_ref.lon == new_ref.lon and old_ref.alt == new_ref.alt:
        return False

    # t_old_origin_in_new = position of old_ref expressed in ENU(new_ref)
    # (same convention as transform_state_between_refs: point=old_ref, ref=new_ref)
    t_old_origin_in_new = geodetic_to_enu(
        old_ref.lat, old_ref.lon, old_ref.alt, new_ref.lat, new_ref.lon, new_ref.alt
    )

    drift = math.sqrt(
        t_old_origin_in_new[0] ** 2 + t_old_origin_in_new[1] ** 2 + t_old_origin_in_new[2] ** 2
    )

    if drift < RECENTER_THRESH_M:
        return False

    # Perform recentering transformation
    R = enu_rotation(old_ref.lat, old_ref.lon, new_ref.lat, new_ref.lon)

    kf = tracks[tid]
    x6 = kf.get_state()  # [pos_old; vel_old] in ENU(old_ref)
    P6 = kf.get_covariance()

    # Transform state: p_new = R * p_old + t_old_origin_in_new
    pos_new = R @ x6[:3] + t_old_origin_in_new  # Position: rotate + translate
    vel_new = R @ x6[3:6]  # Velocity: rotate only

    P_new = _rotate_covariance_blocks(P6, R)

    # Update filter with transformed state
    kf.set_state(np.concatenate([pos_new, vel_new]), P_new)
    track_refs[tid] = new_ref.copy()

    return True


def get_velocity_for_pn(
    tid: int, track_refs: dict, tracks: dict, missile_ref: Position
) -> np.ndarray | None:
    """
    Get target velocity rotated from track ENU to missile ENU (for proportional navigation).

    Args:
        tid: Track ID
        track_refs: Dictionary of track references
        tracks: Dictionary of track filters
        missile_ref: Missile's ENU reference position

    Returns:
        Target velocity in missile's ENU frame, or None if track not found
    """
    if tid not in tracks or tid not in track_refs:
        return None

    v_track = tracks[tid].get_state()[3:6]  # Velocity in track's ENU
    track_ref = track_refs[tid]

    # Rotate from track ENU to missile ENU
    R = enu_rotation(track_ref.lat, track_ref.lon, missile_ref.lat, missile_ref.lon)
    return R @ v_track


def get_target_state_for_pn(tid: int, track_refs: dict, tracks: dict, missile_ref: Position):
    """
    Return (p_tgt_msl, v_tgt_msl, P6_msl) with all quantities in the missile's ENU frame.

    p_tgt_msl : (3,) ndarray  — position of target relative to missile ENU origin
    v_tgt_msl : (3,) ndarray  — velocity of target in missile ENU
    P6_msl    : (6,6) ndarray or None — full position+velocity covariance in missile ENU,
                propagated through the ENU-to-ENU rotation; None if the filter does not
                supply a covariance.

    This is the single authoritative guidance-state export: callers must not mix this
    with raw state[3:6] or unchecked provider velocities.
    """
    out = get_state_in_ref(tid, track_refs, tracks, missile_ref)
    if out is None:
        return None
    x6_msl, P6_msl = out
    return x6_msl[:3], x6_msl[3:6], P6_msl


def transform_state_between_refs(
    state6: np.ndarray, cov6: np.ndarray | None, from_ref: Position, to_ref: Position
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Transform [x,y,z,vx,vy,vz] (and optional 6x6 covariance) from ENU(from_ref) to ENU(to_ref).

    Convention (unambiguous):
        p_to = R * p_from + t_from_origin_in_to
        v_to = R * v_from
        P_to = S * P_from * S^T,  S = blkdiag(R, R)

    where:
        R                   : rotation matrix ENU(from_ref) -> ENU(to_ref)
        t_from_origin_in_to : position of the FROM-frame origin expressed in ENU(to_ref)
                              = geodetic_to_enu(from_ref, ref=to_ref)

    Intuition: R @ p_from gives the displacement from_ref→target in to_ref orientation;
    adding t_from_origin_in_to (which is to_ref→from_ref in to_ref coords) yields the
    displacement to_ref→target, i.e. the target position in ENU(to_ref). ✓
    """
    if from_ref.lat == to_ref.lat and from_ref.lon == to_ref.lon and from_ref.alt == to_ref.alt:
        x_same = np.asarray(state6[:6], dtype=float).copy()
        return (x_same, None) if cov6 is None else (x_same, np.array(cov6, dtype=float))

    # t_from_origin_in_to = position of from_ref expressed in ENU(to_ref)
    # NOTE: arguments are (point=from_ref, ref=to_ref) — NOT the other way around.
    R, t_from_origin_in_to = _frame_rotation_translation(
        from_ref.lat, from_ref.lon, from_ref.alt, to_ref.lat, to_ref.lon, to_ref.alt
    )

    p_from = np.asarray(state6[:3], dtype=float)
    v_from = np.asarray(state6[3:6], dtype=float)
    p_to = R @ p_from + t_from_origin_in_to
    v_to = R @ v_from
    x_to = np.concatenate([p_to, v_to])

    if cov6 is None:
        return x_to, None

    return x_to, _rotate_covariance_blocks(cov6, R)


def transform_states_between_refs_batch(
    states: list[np.ndarray],
    covs: list[np.ndarray],
    from_refs: list[Position],
    to_ref: Position,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized form of :func:`transform_state_between_refs` over N tracks that
    share a single destination ENU(``to_ref``) but each have their own source
    ENU(``from_refs[i]``).

    Returns ``(X_to, P_to)`` with shapes ``(N, 6)`` and ``(N, 6, 6)``. Equals the
    per-track scalar transform to machine epsilon (the column-independent rotation
    and the ``S P S^T`` block rotation are identical math, just stacked). Identity
    rows (``from == to``) use ``R = I, t = 0`` so they reproduce the scalar
    early-out exactly.

    Batching pays here even at the small per-sensor track counts (~2-3): unlike the
    KF update there is no inverse and the scalar path reallocates ``S`` every call,
    so the break-even is below N=2.
    """
    N = len(from_refs)
    X = np.asarray(states, dtype=float).reshape(N, 6)
    P = np.asarray(covs, dtype=float).reshape(N, 6, 6)

    # Shared network pictures are exported from one stable team frame. Avoid
    # expanding that same rotation and block matrix N times for every receiver.
    # The generic path below remains necessary for tracker exports, whose tracks
    # may have different recenter references.
    first_ref = from_refs[0] if N else None
    if first_ref is not None and all(
        ref.lat == first_ref.lat and ref.lon == first_ref.lon and ref.alt == first_ref.alt
        for ref in from_refs[1:]
    ):
        if (
            first_ref.lat == to_ref.lat
            and first_ref.lon == to_ref.lon
            and first_ref.alt == to_ref.alt
        ):
            return X.copy(), P.copy()

        R, t = _frame_rotation_translation(
            first_ref.lat,
            first_ref.lon,
            first_ref.alt,
            to_ref.lat,
            to_ref.lon,
            to_ref.alt,
        )
        X_to = np.empty_like(X)
        X_to[:, :3] = X[:, :3] @ R.T + t
        X_to[:, 3:] = X[:, 3:] @ R.T
        S = np.zeros((6, 6), dtype=float)
        S[:3, :3] = R
        S[3:, 3:] = R
        return X_to, S @ P @ S.T

    R = np.empty((N, 3, 3), dtype=float)
    t = np.zeros((N, 3), dtype=float)
    for i, fr in enumerate(from_refs):
        if fr.lat == to_ref.lat and fr.lon == to_ref.lon and fr.alt == to_ref.alt:
            R[i] = np.eye(3)  # t already 0 -> exact identity, matches scalar early-out
        else:
            R_i, t_i = _frame_rotation_translation(
                fr.lat, fr.lon, fr.alt, to_ref.lat, to_ref.lon, to_ref.alt
            )
            R[i] = R_i
            t[i] = t_i

    p_to = np.einsum("nij,nj->ni", R, X[:, :3]) + t
    v_to = np.einsum("nij,nj->ni", R, X[:, 3:6])
    X_to = np.concatenate([p_to, v_to], axis=1)

    # S = blkdiag(R, R); P_to = S P S^T (single batched 6x6, matching _rotate_covariance_blocks)
    S = np.zeros((N, 6, 6), dtype=float)
    S[:, :3, :3] = R
    S[:, 3:, 3:] = R
    P_to = S @ P @ S.transpose(0, 2, 1)
    return X_to, P_to


def get_state_in_ref(
    tid: int, track_refs: dict, tracks: dict, export_ref: Position
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Read track state+cov and transform them into ENU(export_ref).
    """
    if tid not in tracks or tid not in track_refs:
        return None
    x6 = tracks[tid].get_state()
    P6 = tracks[tid].get_covariance()
    from_ref = track_refs[tid]
    return transform_state_between_refs(x6, P6, from_ref, export_ref)
