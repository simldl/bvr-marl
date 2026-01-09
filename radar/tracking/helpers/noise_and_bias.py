# noise_and_bias.py

import math
import numpy as np

from radar.tracking.filter.filters import ConstantVelocityKFFilter

Z_BIAS_ALPHA = 0.02

def apply_anisotropic_R(self, meta, tracker: ConstantVelocityKFFilter, meas_enu, cluster):
    """
    Set measurement covariance on the tracker, based on the *source* of the measurement*.

    - Datalink/GT ('T' present): tight, isotropic R (we trust it).
    - Radar: use SENSOR GEOMETRY (cluster['d'], 'az', 'el'), not the track ENU radius.

    This avoids the previous sensitivity of R to recentering choice.
    """
    # If datalink / GT present, trust it more.
    if cluster.get('T') is not None and hasattr(cluster['T'], "position"):
        tracker.set_measurement_std((5.0, 5.0, 5.0))
        meta['meas_source'] = 'datalink'
        return

    # --- Radar-only noise model (anisotropic) ---
    # Pull range/angles from the cluster; fall back safely if missing.
    d = float(cluster.get('d', np.hypot(float(meas_enu[0]), float(meas_enu[1]))))
    el_deg = float(cluster.get('el', 0.0))
    # Sensor/angular resolution parameters (configurable on TrackerManager)
    range_res_m = float(getattr(self, "range_resolution_m", 150.0))
    ang_res_deg = float(getattr(self, "angular_resolution_deg", 2.0))
    ang_res_rad = math.radians(ang_res_deg)

    # Lateral standard deviation scales with angular resolution and range
    sigma_xy = max(0.5 * range_res_m, 0.5 * ang_res_rad * d)
    # Vertical std can be a bit tighter; optionally use elevation geometry too
    # (keeping your previous conservative floor)
    sigma_z = max(10.0, 0.25 * range_res_m)

    # Inflate noise under deception (ECM ghosts have higher uncertainty)
    if cluster.get('is_deception', False):
        sigma_xy *= 1.5  # 50% inflation in lateral
        sigma_z *= 1.2   # 20% inflation in vertical

    tracker.set_measurement_std((sigma_xy, sigma_xy, sigma_z))
    meta['meas_source'] = 'radar'


def apply_z_bias_correction(meta: dict, tracker: ConstantVelocityKFFilter, 
                           meas_enu: np.ndarray) -> np.ndarray:
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
    z_pred = float(tracker.get_state()[2])  # Predicted altitude
    innov_z = float(meas_enu[2] - z_pred)   # Innovation in z
    
    # Update bias estimate using EWMA
    alpha = float(Z_BIAS_ALPHA)
    current_bias = float(meta.get('z_bias', 0.0))
    meta['z_bias'] = (1.0 - alpha) * current_bias + alpha * innov_z
    
    # Apply bias correction
    meas_adj = np.array(meas_enu, dtype=float)
    meas_adj[2] = meas_adj[2] - meta['z_bias']
    
    return meas_adj

def apply_xy_bias_correction(meta: dict, tracker: ConstantVelocityKFFilter,
                             meas_enu: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Track and remove a small fixed lateral (E/N) bias via EWMA of the innovations.
    Very small alpha so we don't "learn" maneuvers.
    """
    x_pred = tracker.get_state()
    ex = float(meas_enu[0] - x_pred[0])
    ey = float(meas_enu[1] - x_pred[1])

    bx, by = meta.get('xy_bias', (0.0, 0.0))
    bx = (1.0 - alpha) * float(bx) + alpha * ex
    by = (1.0 - alpha) * float(by) + alpha * ey
    meta['xy_bias'] = (bx, by)

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
    return {
        'range_resolution_m': 150.0,
        'angular_resolution_deg': 2.0
    }