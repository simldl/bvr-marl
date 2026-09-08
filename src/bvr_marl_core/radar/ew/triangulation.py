"""Cross-radar triangulation of jammer bearing strobes.

A single radar that is being noise-jammed measures only a bearing to the jammer
(its range is denied). Two or more datalinked radars observing the same jammer from
sufficiently different aspects can intersect their bearing lines to recover the
jammer's 3-D position, defeating the jam cooperatively. This module solves the
least-squares closest point to a set of 3-D bearing lines.
"""

import math

import numpy as np

# Minimum angle (deg) between two bearing lines for the intersection to be well
# conditioned. Below this the observers see the jammer from nearly the same aspect
# and the range estimate is unreliable, so we fall back to bearing-only.
MIN_BASELINE_ANGLE_DEG = 8.0


def triangulate_pair_normalized(p0: np.ndarray, u0: np.ndarray, p1: np.ndarray, u1: np.ndarray):
    """Allocation-light closest point for two already-normalized bearing rays."""
    cosine = max(-1.0, min(1.0, float(u0 @ u1)))
    if abs(cosine) > math.cos(math.radians(MIN_BASELINE_ANGLE_DEG)):
        return None, False
    offset = p0 - p1
    d0 = float(u0 @ offset)
    d1 = float(u1 @ offset)
    denominator = 1.0 - cosine * cosine
    if denominator <= 1e-12:
        return None, False
    along0 = (cosine * d1 - d0) / denominator
    along1 = (d1 - cosine * d0) / denominator
    point = 0.5 * ((p0 + along0 * u0) + (p1 + along1 * u1))
    if not all(math.isfinite(float(value)) for value in point):
        return None, False
    return point, True


def pairwise_bearing_candidates(observers: np.ndarray, directions: np.ndarray):
    """Vectorized all-pairs analog of :func:`triangulate_pair_normalized` plus the
    forward/residual gate used by strobe association.

    ``observers`` is ``(S, 3)`` and ``directions`` is ``(S, 3)`` **unit** bearing
    vectors. Returns ``(residual, accept)``, each ``(S, S)``, defined only on the
    strict upper triangle ``i < j`` (diagonal and lower triangle are ``False`` /
    ``inf``). ``accept[i, j]`` is True exactly when the scalar path would have
    produced a candidate for that pair: well-conditioned baseline, both rays
    pointing forward, and angular residual within tolerance. Matches the per-pair
    scalar computation to floating-point precision — it is the same closed form,
    evaluated over the grid instead of in a Python double loop.
    """
    positions = np.ascontiguousarray(observers, dtype=float)  # (S, 3)
    unit = np.ascontiguousarray(directions, dtype=float)  # (S, 3), assumed normalized
    count = positions.shape[0]

    cosine = np.clip(unit @ unit.T, -1.0, 1.0)  # (S, S)
    # up[i, j] = unit[i] . positions[j]; diagonal is unit[i] . positions[i].
    up = unit @ positions.T
    diag_up = np.diagonal(up)
    d0 = diag_up[:, None] - up  # unit[i] . (positions[i] - positions[j])
    d1 = up.T - diag_up[None, :]  # unit[j] . (positions[i] - positions[j])
    denom = 1.0 - cosine * cosine

    with np.errstate(divide="ignore", invalid="ignore"):
        along0 = (cosine * d1 - d0) / denom
        along1 = (d1 - cosine * d0) / denom
        point = 0.5 * (
            positions[:, None, :]
            + along0[..., None] * unit[:, None, :]
            + positions[None, :, :]
            + along1[..., None] * unit[None, :, :]
        )  # (S, S, 3), the midpoint of the two ray-closest points
        offset0 = point - positions[:, None, :]
        offset1 = point - positions[None, :, :]
        forward0 = np.einsum("ijk,ik->ij", offset0, unit)  # (S, S)
        forward1 = np.einsum("ijk,jk->ij", offset1, unit)
        perp0 = np.linalg.norm(offset0 - forward0[..., None] * unit[:, None, :], axis=2)
        perp1 = np.linalg.norm(offset1 - forward1[..., None] * unit[None, :, :], axis=2)
        residual = perp0 / forward0 + perp1 / forward1

    cosine_gate = math.cos(math.radians(MIN_BASELINE_ANGLE_DEG))
    residual_gate = 2.0 * math.sin(math.radians(5.0))
    accept = (
        (np.abs(cosine) <= cosine_gate)
        & (denom > 1e-12)
        & np.isfinite(point).all(axis=2)
        & (forward0 > 0.0)
        & (forward1 > 0.0)
        & (residual <= residual_gate)
    )
    accept &= np.triu(np.ones((count, count), dtype=bool), k=1)  # keep i < j only
    residual = np.where(accept, residual, np.inf)
    return residual, accept


def triangulate(observers_enu: list[np.ndarray], directions_enu: list[np.ndarray]):
    """Least-squares intersection of bearing lines.

    Each line i is {p_i + s * u_i}. Returns (point_enu, ok) where ``ok`` is False
    when there are fewer than two lines or the geometry is ill-conditioned (nearly
    parallel bearings); ``point_enu`` is None in that case.
    """
    if len(observers_enu) < 2:
        return None, False
    if len(observers_enu) == 2:
        u0 = directions_enu[0] / (np.linalg.norm(directions_enu[0]) + 1e-12)
        u1 = directions_enu[1] / (np.linalg.norm(directions_enu[1]) + 1e-12)
        return triangulate_pair_normalized(observers_enu[0], u0, observers_enu[1], u1)

    positions = np.asarray(observers_enu, dtype=float)  # (k, 3)
    raw = np.asarray(directions_enu, dtype=float)  # (k, 3)

    # Reject poor geometry: require some pair of bearings to differ enough. The
    # widest pair angle corresponds to the smallest pairwise cosine (acos is
    # monotone), so one acos on that minimum reproduces the max over all pairs.
    cosines = np.clip(raw @ raw.T, -1.0, 1.0)
    off_diagonal = cosines[np.triu_indices(len(raw), k=1)]
    max_angle = math.degrees(math.acos(float(off_diagonal.min())))
    if max_angle < MIN_BASELINE_ANGLE_DEG:
        return None, False

    # Minimise sum_i || (I - u_i u_i^T)(x - p_i) ||^2  ->  A x = b, with
    # A = k*I - U^T U and b = sum_i p_i - U^T (u_i . p_i).
    unit = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12)
    A = len(unit) * np.eye(3) - unit.T @ unit
    along = np.einsum("ij,ij->i", unit, positions)
    b = positions.sum(axis=0) - unit.T @ along
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, False
    if not np.all(np.isfinite(x)):
        return None, False
    return x, True
