import math

import numpy as np

from bvr_marl_core.radar.tracking.filter.filters import ConstantVelocityKFFilter
from bvr_marl_core.radar.tracking.helpers.recenter_logic import transform_state_between_refs
from bvr_marl_core.simulator.core.helpers import Position

N_MAX_RADARS = 3


def spawn_tracker(
    self, c, meas: np.ndarray, dt: float, ref_pos: Position
) -> ConstantVelocityKFFilter:
    """
    Spawn a CV KF and seed its initial covariance from the *first* measurement type.
    """
    kf = ConstantVelocityKFFilter(meas, dt, process_noise_std=50.0, measurement_noise_var=1.0)

    if c.get("T") is not None and hasattr(c["T"], "position"):
        sx, sy, sz = 20.0, 20.0, 20.0
    else:
        range_res_m = float(getattr(self, "range_resolution_m", 150.0))
        ang_res_deg = float(getattr(self, "angular_resolution_deg", 2.0))
        ang_res_rad = math.radians(ang_res_deg)
        # Prefer the sensor-measured slant range (cluster['d']) over the ENU-derived
        # distance: meas may be in a transformed reference frame (datalink, recenter).
        sensor_range = c.get("d", None)
        if sensor_range is not None and float(sensor_range) > 0.0:
            rng_xy = float(sensor_range)
        else:
            rng_xy = float(np.hypot(meas[0], meas[1])) + 1e-6
        sx = max(0.5 * range_res_m, 0.5 * ang_res_rad * rng_xy)
        sy = sx
        sz = max(10.0, 0.25 * range_res_m)

    R0 = np.diag([sx * sx, sy * sy, sz * sz])
    P0 = np.zeros((6, 6), dtype=float)
    P0[:3, :3] = np.maximum(R0, np.diag([25.0, 25.0, 25.0]))  # ≥ 5 m std on position
    P0[3:, 3:] = np.diag([90000.0, 90000.0, 90000.0])  # (300 m/s)^2 on velocity

    kf.set_state(np.concatenate([meas.astype(float), np.zeros(3, dtype=float)]), P0)
    kf.set_measurement_std((sx, sy, sz))
    return kf


def prune_tracks(tracks: dict, track_refs: dict, track_meta: dict, timeout: int = 5):
    """
    Remove tracks that haven't been updated for too long.

    Args:
        tracks: Dictionary of track filters
        track_refs: Dictionary of track references
        track_meta: Dictionary of track metadata
        timeout: Maximum number of missed updates before pruning
    """
    remove = []
    for tid, kf in tracks.items():
        if getattr(kf, "missed_updates", 0) >= timeout:
            remove.append(tid)

    for tid in remove:
        tracks.pop(tid, None)
        track_refs.pop(tid, None)
        track_meta.pop(tid, None)


def build_track_outputs(
    tracks: dict,
    track_refs: dict,
    track_meta: dict,
    clusters: list,
    export_ref: Position | None = None,
) -> list[tuple]:
    """
    Build output tuples for all tracks.
    If export_ref is provided, states/covariances are rotated into that ENU.
    """
    outs = []
    for tid, kf in tracks.items():
        meta = track_meta[tid]
        x6 = kf.get_state()
        P6 = kf.get_covariance()
        ref = track_refs.get(tid)

        if export_ref is not None and ref is not None:
            x6, P6 = transform_state_between_refs(x6, P6, ref, export_ref)
            out_ref = export_ref  # make it explicit the frame changed
        else:
            out_ref = ref

        # Gather context
        n_obs = meta.get("n_obs_hist", [1])[-1]
        lifetime = meta.get("lifetime", 1)
        update_count = meta.get("update_count", 0)
        confidence = calculate_confidence(meta, kf)

        # ECM fields
        is_deception = meta.get("is_deception", False)
        suspect_deception = meta.get("suspect_deception", False)
        engagement_id = meta.get("engagement_id", None)
        jammer_id = meta.get("jammer_id", None)

        # --- Recover concrete target (if any) for engageability/exports ---
        # Prefer whatever the tracker stored in metadata
        tgt = meta.get("tgt", None)
        utype = meta.get("unit_type", meta.get("utype", None))
        if tgt is None and clusters:
            # Heuristic: for single-target scenarios, clusters often carry 'T'
            # Pick the first cluster with a concrete unit handle.
            for cl in clusters:
                if isinstance(cl, dict) and cl.get("T") is not None:
                    tgt = cl["T"]
                    if utype is None:
                        utype = getattr(tgt, "unit_type", None)
                    break

        engageable = ((tgt is not None) or (engagement_id is not None)) and (not suspect_deception)

        outs.append(
            (
                tid,
                x6,
                P6[:3, :3],
                tgt,
                utype,
                out_ref,
                confidence,
                n_obs,
                lifetime,
                update_count,
                is_deception,
                suspect_deception,
                engagement_id,
                jammer_id,
                engageable,
            )
        )
    return outs


def calculate_confidence(meta: dict, kf) -> float:
    """
    Calculate track confidence from five orthogonal factors.

    confidence = clamp01(
        (0.60 * obs_score + 0.40 * maturity) * recency * consistency * accuracy
    )

    Factors
    -------
    obs_score   : EMA of per-update sensor-hit count, normalised by N_MAX_RADARS.
                  Rewards tracks seen by multiple radars / datalink.
    maturity    : Fraction of MATURITY_THRESH (10) successful updates reached.
                  New tentative tracks start near 0 and grow linearly.
    recency     : exp(-k * missed_updates), k = ln(2) ≈ 0.693.
                  Halves per missed update at integer values but varies smoothly.
    consistency : exp(-max(0, nis_ema - expected) / expected).
                  NIS_EMA initialised to 3.0 (chi² DOF=3); exponential penalty
                  is smoother than the old reciprocal and decays faster at high NIS,
                  which is better for deception / mis-association detection.
    accuracy    : Rational falloff 1/(1+(pos_std/SIGMA_REF)²) with SIGMA_REF=500 m.
                  Avoids arbitrary hard endpoints; half-power at 500 m position std.
    """
    # --- obs_score ---
    obs_ema = meta.get("obs_ema", 1.0)
    obs_score = min(1.0, obs_ema / N_MAX_RADARS)

    # --- maturity ---
    MATURITY_THRESH = 10
    updates_with_meas = meta.get("updates_with_meas", 0)
    maturity = min(1.0, updates_with_meas / MATURITY_THRESH)

    # --- recency: exp(-ln2 * missed) ≡ 0.5^missed at integers, smooth otherwise ---
    missed = getattr(kf, "missed_updates", 0)
    _LN2 = 0.6931471805599453
    recency = math.exp(-_LN2 * missed)

    # --- consistency: exponential NIS penalty (smoother than reciprocal) ---
    NIS_EXPECTED = 3.0
    nis_ema = meta.get("nis_ema", NIS_EXPECTED)
    consistency = math.exp(-max(0.0, nis_ema - NIS_EXPECTED) / NIS_EXPECTED)

    # --- accuracy: rational 1/(1+(sigma/SIGMA_REF)^2), SIGMA_REF=500 m ---
    try:
        P = kf.get_covariance()
        pos_var_rms = (P[0, 0] + P[1, 1] + P[2, 2]) / 3.0
        pos_std = math.sqrt(max(0.0, pos_var_rms))
    except Exception:
        pos_std = 5000.0  # safe pessimistic default
    SIGMA_REF = 500.0
    accuracy = 1.0 / (1.0 + (pos_std / SIGMA_REF) ** 2)

    raw = (0.60 * obs_score + 0.40 * maturity) * recency * consistency * accuracy
    return max(0.0, min(1.0, raw))


def get_or_create_meta(track_meta: dict, tid: int) -> dict:
    """
    Get metadata for track ID, creating default if not exists.

    Args:
        track_meta: Track metadata dictionary
        tid: Track ID

    Returns:
        Metadata dictionary for the track
    """
    if tid not in track_meta:
        track_meta[tid] = {
            "n_obs_hist": [],
            "lifetime": 0,
            "update_count": 0,
            "last_meas": None,
            "last_dt": None,
            "z_bias": 0.0,
            "xy_bias": (0.0, 0.0),
            # ECM fields
            "is_deception": False,
            "suspect_deception": False,
            "engagement_id": None,
            "jammer_id": None,
            # Confidence v2 fields
            "obs_ema": 0.0,
            "updates_with_meas": 0,
            "nis_ema": 3.0,  # initialised to chi2 expected value (DOF=3)
        }
    return track_meta[tid]
