import math
import numpy as np
from typing import List, Tuple

from simulator.core.helpers import Position
from radar.tracking.filter.filters import ConstantVelocityKFFilter
from radar.tracking.helpers.recenter_logic import transform_state_between_refs

N_MAX_RADARS = 3


def spawn_tracker(self, c, meas: np.ndarray, dt: float, ref_pos: Position) -> ConstantVelocityKFFilter:
    """
    Spawn a CV KF and seed its initial covariance from the *first* measurement type.
    """
    # Create with a reasonable process noise (sigma_a ~ 50 m/s^2 by default here)
    kf = ConstantVelocityKFFilter(meas, dt, process_noise_std=50.0, measurement_noise_var=1.0)

    # Build an R equivalent to what _apply_anisotropic_R would have set, then
    # seed P (position) from that R and give a large initial velocity variance.
    if c.get('T') is not None and hasattr(c['T'], "position"):
        sx, sy, sz = 20.0, 20.0, 20.0
    else:
        range_res_m = float(getattr(self, "range_resolution_m", 150.0))
        ang_res_deg = float(getattr(self, "angular_resolution_deg", 2.0))
        ang_res_rad = math.radians(ang_res_deg)
        rng_xy = float(np.hypot(meas[0], meas[1])) + 1e-6
        sx = max(0.5 * range_res_m, 0.5 * ang_res_rad * rng_xy)
        sy = sx
        sz = max(10.0, 0.25 * range_res_m)

    R0 = np.diag([sx*sx, sy*sy, sz*sz])
    P0 = np.zeros((6, 6), dtype=float)
    P0[:3, :3] = np.maximum(R0, np.diag([25.0, 25.0, 25.0]))  # ≥ 5 m std on position
    P0[3:, 3:] = np.diag([90000.0, 90000.0, 90000.0])         # (300 m/s)^2 on velocity

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
        if getattr(kf, 'missed_updates', 0) >= timeout:
            remove.append(tid)
    
    for tid in remove:
        tracks.pop(tid, None)
        track_refs.pop(tid, None)
        track_meta.pop(tid, None)


def build_track_outputs(tracks: dict, track_refs: dict, track_meta: dict,
                        clusters: List, export_ref: Position | None = None) -> List[Tuple]:
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
        n_obs = meta.get('n_obs_hist', [1])[-1]
        lifetime = meta.get('lifetime', 1)
        update_count = meta.get('update_count', 0)
        confidence = calculate_confidence(meta.get('n_obs_hist', [1]), lifetime)

        # ECM fields
        is_deception = meta.get('is_deception', False)
        suspect_deception = meta.get('suspect_deception', False)
        engagement_id = meta.get('engagement_id', None)
        jammer_id = meta.get('jammer_id', None)

        # --- Recover concrete target (if any) for engageability/exports ---
        # Prefer whatever the tracker stored in metadata
        tgt   = meta.get('tgt', None)
        utype = meta.get('unit_type', meta.get('utype', None))
        if tgt is None and clusters:
            # Heuristic: for single-target scenarios, clusters often carry 'T'
            # Pick the first cluster with a concrete unit handle.
            for cl in clusters:
                if isinstance(cl, dict) and cl.get('T') is not None:
                    tgt = cl['T']
                    if utype is None:
                        utype = getattr(tgt, 'unit_type', None)
                    break

        engageable = ((tgt is not None) or (engagement_id is not None)) and (not suspect_deception)


        # Optional: find target object
        tgt = None
        for c in clusters:
            c_tid = getattr(c.get('T', None), 'id', None) if c.get('T', None) is not None else engagement_id
            if c_tid == tid or (engagement_id is not None and c_tid == engagement_id):
                tgt = c.get('T', None)
                break
        utype = type(tgt).__name__ if tgt is not None else None

        outs.append((tid, x6, P6[:3, :3], tgt, utype, out_ref,
                     confidence, n_obs, lifetime, update_count,
                     is_deception, suspect_deception, engagement_id, jammer_id, engageable))
    return outs


def calculate_confidence(n_obs_hist: List[int], lifetime: int) -> float:
    """
    Calculate track confidence based on observation history and lifetime.
    
    Args:
        n_obs_hist: History of number of observations per update
        lifetime: Track lifetime in number of updates
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    n_obs_mean = np.mean(n_obs_hist)
    obs_factor = min(1.0, n_obs_mean / N_MAX_RADARS)
    life_factor = min(1.0, math.log1p(lifetime) / 5.0)
    return 0.7 * obs_factor + 0.3 * life_factor


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
            'n_obs_hist': [],
            'lifetime': 0,
            'update_count': 0,
            'last_meas': None,
            'last_dt': None,
            'z_bias': 0.0,
            'xy_bias': (0.0, 0.0),
            # ECM fields
            'is_deception': False,
            'suspect_deception': False,
            'engagement_id': None,
            'jammer_id': None,
        }
    return track_meta[tid]