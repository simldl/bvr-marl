"""Observation normalization helpers.

Scales raw physical quantities (meters, m/s) into roughly [-1, 1] so the policy
network sees well-conditioned inputs. These are applied only when building the
RL observation vector — the underlying geometry helpers (rel_state, etc.) stay
in physical units so scripted BT controllers are unaffected.
"""

import math

import numpy as np

# Reference scales. Values beyond these saturate at +/-1 (far/fast entities are
# treated as "far"/"fast" without needing to distinguish exact magnitudes).
OBS_POS_REF_M = 150_000.0  # horizontal relative range (~radar/engagement scale)
OBS_ALT_REF_M = 15_000.0  # vertical relative range (altitude difference)
OBS_VEL_REF_MPS = 1000.0  # relative velocity (covers fighter + missile closure)
OBS_RANGE_REF_M = OBS_POS_REF_M  # scalar range/DLZ/NEZ features use the same engagement scale


def to_body_frame(vec6, yaw_deg) -> list[float]:
    """Rotate an ENU relative pos/vel 6-vector into the ownship body frame.

    Input [dE, dN, dU, dvE, dvN, dvU] (East/North/Up). Output
    [forward, right, up, v_forward, v_right, v_up] relative to the ownship nose,
    so "enemy ahead and to the right" is directly represented instead of needing
    the network to combine world-frame deltas with the ownship heading.

    Geographic yaw (0=N, 90=E): heading dir = (sin y, cos y) in (E, N);
    right-of-heading dir = (cos y, -sin y).
    """
    dE, dN, dU, dvE, dvN, dvU = (float(x) for x in vec6[:6])
    y = math.radians(float(yaw_deg))
    sy, cy = math.sin(y), math.cos(y)
    fwd = dE * sy + dN * cy
    right = dE * cy - dN * sy
    v_fwd = dvE * sy + dvN * cy
    v_right = dvE * cy - dvN * sy
    return [fwd, right, dU, v_fwd, v_right, dvU]


def normalize_pos_vel(vec6) -> list[float]:
    """Normalize a 6-vector [dE, dN, dU, dvE, dvN, dvU] to clipped [-1, 1].

    Horizontal position uses OBS_POS_REF_M, vertical position OBS_ALT_REF_M, and
    all three velocity components OBS_VEL_REF_MPS.
    """
    dE, dN, dU, dvE, dvN, dvU = (float(x) for x in vec6[:6])
    return [
        float(np.clip(dE / OBS_POS_REF_M, -1.0, 1.0)),
        float(np.clip(dN / OBS_POS_REF_M, -1.0, 1.0)),
        float(np.clip(dU / OBS_ALT_REF_M, -1.0, 1.0)),
        float(np.clip(dvE / OBS_VEL_REF_MPS, -1.0, 1.0)),
        float(np.clip(dvN / OBS_VEL_REF_MPS, -1.0, 1.0)),
        float(np.clip(dvU / OBS_VEL_REF_MPS, -1.0, 1.0)),
    ]


def normalize_range_m(value_m) -> float:
    """Normalize a non-negative engagement range in meters to [0, 1]."""
    try:
        value = float(value_m)
    except (TypeError, ValueError):
        value = 0.0
    return float(np.clip(value / OBS_RANGE_REF_M, 0.0, 1.0))
