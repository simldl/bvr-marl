import math

import numpy as np

from bvr_marl_core.radar.tracking.filter.filters import ConstantVelocityKFFilter

Z_BIAS_ALPHA = 0.02


def apply_anisotropic_R(self, meta, tracker: ConstantVelocityKFFilter, meas_enu, cluster):
    """
    Set measurement covariance on the tracker, based on the *source* of the measurement*.

    - Datalink/GT ('T' present): tight, isotropic R (we trust it).
    - Radar: use SENSOR GEOMETRY (cluster['d'], 'az', 'el'), not the track ENU radius.

    This avoids the previous sensitivity of R to recentering choice.
    """
    if cluster.get("T") is not None and hasattr(cluster["T"], "position"):
        tracker.set_measurement_std((5.0, 5.0, 5.0))
        meta["meas_source"] = "datalink"
        return

    d = float(cluster.get("d", np.hypot(float(meas_enu[0]), float(meas_enu[1]))))
    range_res_m = float(getattr(self, "range_resolution_m", 150.0))
    ang_res_rad = math.radians(float(getattr(self, "angular_resolution_deg", 2.0)))

    sigma_xy = max(0.5 * range_res_m, 0.5 * ang_res_rad * d)
    sigma_z = max(10.0, 0.25 * range_res_m)

    if cluster.get("is_deception", False):
        sigma_xy *= 1.5
        sigma_z *= 1.2

    tracker.set_measurement_std((sigma_xy, sigma_xy, sigma_z))
    meta["meas_source"] = "radar"


def apply_z_bias_correction(
    meta: dict, tracker: ConstantVelocityKFFilter, meas_enu: np.ndarray
) -> np.ndarray:
    """
    Apply z-bias correction using exponentially weighted moving average (EWMA).

    Compensates for systematic radar altitude bias by tracking the
    innovation sequence and applying a learned bias correction.

    Args:
        meta: Track metadata dictionary containing bias state
        tracker: Kalman filter for prediction
        meas_enu: Raw measurement in ENU coordinates

    Returns:
        Bias-corrected measurement
    """
    z_pred = float(tracker.get_state()[2])
    innov_z = float(meas_enu[2] - z_pred)

    alpha = float(Z_BIAS_ALPHA)
    current_bias = float(meta.get("z_bias", 0.0))
    meta["z_bias"] = (1.0 - alpha) * current_bias + alpha * innov_z

    meas_adj = np.array(meas_enu, dtype=float)
    meas_adj[2] = meas_adj[2] - meta["z_bias"]

    return meas_adj


def apply_xy_bias_correction(
    meta: dict, tracker: ConstantVelocityKFFilter, meas_enu: np.ndarray, alpha: float = 0.01
) -> np.ndarray:
    """
    Track and remove a small fixed lateral (E/N) bias via EWMA of the innovations.
    Very small alpha so we don't "learn" maneuvers.
    """
    x_pred = tracker.get_state()
    ex = float(meas_enu[0] - x_pred[0])
    ey = float(meas_enu[1] - x_pred[1])

    bx, by = meta.get("xy_bias", (0.0, 0.0))
    bx = (1.0 - alpha) * float(bx) + alpha * ex
    by = (1.0 - alpha) * float(by) + alpha * ey
    meta["xy_bias"] = (bx, by)

    meas_adj = np.array(meas_enu, dtype=float)
    meas_adj[0] -= bx
    meas_adj[1] -= by
    return meas_adj


def get_default_noise_parameters() -> dict:
    """
    Get default noise model parameters.

    Returns:
        Dictionary with default range and angular resolution values
    """
    return {"range_resolution_m": 150.0, "angular_resolution_deg": 2.0}
