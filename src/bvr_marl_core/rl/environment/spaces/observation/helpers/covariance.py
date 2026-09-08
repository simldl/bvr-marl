"""Covariance frame transforms for observation features.

The ownship-body rotation used here matches ``normalization.to_body_frame`` exactly,
so a covariance is expressed in the same frame as the state it belongs to.
"""

from __future__ import annotations

import math

import numpy as np

_COV_EPS = 1e-9


def body_frame_rotation(yaw_deg: float) -> np.ndarray:
    """3x3 rotation mapping an ENU vector to the ownship body frame.

    Matches ``normalization.to_body_frame``:
        forward = dE*sin(y) + dN*cos(y);  right = dE*cos(y) - dN*sin(y);  up = dU.
    """
    y = math.radians(float(yaw_deg))
    sy, cy = math.sin(y), math.cos(y)
    return np.array(
        [
            [sy, cy, 0.0],
            [cy, -sy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def rotate_cov_to_body(cov6, yaw_deg: float) -> np.ndarray:
    """Rotate a 6x6 ENU pos/vel covariance into the body frame: P_body = J P Jᵀ.

    ``J = diag(R, R)`` applies the same rotation to the position and velocity blocks.
    The result is symmetrized and eigenvalue-clamped so it stays positive semidefinite
    after the transform.
    """
    P = np.asarray(cov6, dtype=float)
    if P.shape != (6, 6):
        raise ValueError(f"covariance must be 6x6, got {P.shape}")
    R = body_frame_rotation(yaw_deg)
    J = np.zeros((6, 6), dtype=float)
    J[:3, :3] = R
    J[3:, 3:] = R
    Pb = J @ P @ J.T
    Pb = 0.5 * (Pb + Pb.T)  # symmetrize numerically
    # Clamp eigenvalues to >= eps so downstream sqrt/log are well defined.
    w, V = np.linalg.eigh(Pb)
    w = np.clip(w, _COV_EPS, None)
    return (V * w) @ V.T


__all__ = ["body_frame_rotation", "rotate_cov_to_body"]
