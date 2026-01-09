import numpy as np

from simulator.core.helpers import Position
from radar.core.utils import geodetic_to_enu
from radar.tracking.helpers.enu_utils import enu_rotation


RECENTER_THRESH_M = 1000.0


def maybe_recenter_reference(tid: int, track_refs: dict, tracks: dict, 
                           new_ref: Position) -> bool:
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

    # Calculate drift between old and new reference
    delta = geodetic_to_enu(old_ref.lat, old_ref.lon, old_ref.alt,
                            new_ref.lat, new_ref.lon, new_ref.alt)

    drift = float(np.linalg.norm(delta))
    
    if drift < RECENTER_THRESH_M:
        return False

    # Perform recentering transformation
    R = enu_rotation(old_ref.lat, old_ref.lon, new_ref.lat, new_ref.lon)
    
    kf = tracks[tid]
    x6 = kf.get_state()            # [pos_old; vel_old] in ENU(old_ref)
    P6 = kf.get_covariance()

    # Transform state
    pos_new = R @ x6[:3] + delta   # Position: rotate + translate
    vel_new = R @ x6[3:6]          # Velocity: rotate only
    
    # Transform covariance
    S = np.zeros((6, 6))
    S[:3, :3] = R  # Position block
    S[3:, 3:] = R  # Velocity block
    P_new = S @ P6 @ S.T

    # Update filter with transformed state
    kf.set_state(np.concatenate([pos_new, vel_new]), P_new)
    track_refs[tid] = new_ref.copy()
    
    return True


def get_velocity_for_pn(tid: int, track_refs: dict, tracks: dict, 
                       missile_ref: Position) -> np.ndarray | None:
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

def get_target_state_for_pn(tid: int, track_refs: dict, tracks: dict,
                            missile_ref: Position):
    """
    Return (p_tgt_msl, v_tgt_msl) with both in the missile's ENU frame.
    """
    out = get_state_in_ref(tid, track_refs, tracks, missile_ref)
    if out is None:
        return None
    x6_msl, _ = out
    p_tgt_msl = x6_msl[:3]
    v_tgt_msl = x6_msl[3:6]
    return p_tgt_msl, v_tgt_msl

def transform_state_between_refs(state6: np.ndarray,
                                 cov6: np.ndarray | None,
                                 from_ref: Position,
                                 to_ref: Position) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Transform [x,y,z,vx,vy,vz] (and optional 6x6 covariance) from ENU(from_ref) to ENU(to_ref).

    p_to = R * p_from + Δ
    v_to = R * v_from
    P_to = S * P_from * S^T, where S = blkdiag(R, R)
    """
    R = enu_rotation(from_ref.lat, from_ref.lon, to_ref.lat, to_ref.lon)
    # delta = position of to_ref expressed in ENU(from_ref)
    delta = geodetic_to_enu(from_ref.lat, from_ref.lon, from_ref.alt,
                            to_ref.lat,   to_ref.lon,   to_ref.alt)


    p_from = np.asarray(state6[:3], dtype=float)
    v_from = np.asarray(state6[3:6], dtype=float)
    p_to = R @ p_from + delta
    v_to = R @ v_from
    x_to = np.concatenate([p_to, v_to])

    if cov6 is None:
        return x_to, None

    S = np.zeros((6, 6), dtype=float)
    S[:3, :3] = R
    S[3:, 3:] = R
    P_to = S @ cov6 @ S.T
    return x_to, P_to


def get_state_in_ref(tid: int,
                     track_refs: dict,
                     tracks: dict,
                     export_ref: Position) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Read track state+cov and transform them into ENU(export_ref).
    """
    if tid not in tracks or tid not in track_refs:
        return None
    x6 = tracks[tid].get_state()
    P6 = tracks[tid].get_covariance()
    from_ref = track_refs[tid]
    return transform_state_between_refs(x6, P6, from_ref, export_ref)