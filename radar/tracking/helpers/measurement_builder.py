import numpy as np
from typing import List, Tuple, Any

from simulator.core.helpers import Position
from radar.core.utils import to_cart, geodetic_to_enu



def build_measurements_with_ref(clusters, track_refs: dict, default_ref: Position) -> List[Tuple]:
    """
    Build measurements from clusters, determining ENU reference for each track.

    Returns tuples: (track_id, cluster, measurement_enu, n_obs, used_ref)
    """
    out: List[Tuple[Any, dict, np.ndarray, int, Position]] = []
    for c in clusters:
        tid = get_track_id(c)

        # Choose the ENU reference used for this track (persist per-track)
        ref = track_refs.get(tid)
        if ref is None:
            # Prefer a per-measurement source_pos if present (e.g., remote radar origin),
            # otherwise fall back to the caller’s default_ref
            ref = c.get("source_pos", None) or default_ref
            track_refs[tid] = ref.copy()

        # Build ENU measurement expressed in this 'ref'
        meas_enu = cluster_to_track_measurement(c, ref)
        n_obs = int(c.get('n_obs', 1))
        out.append((tid, c, meas_enu, n_obs, ref.copy()))
    return out


def cluster_to_track_measurement(c, ref: Position) -> np.ndarray:
    # 1) Prefer datalink (GT). c['T'] can be a list -> unwrap it.
    T = c.get('T', None)
    if isinstance(T, list):
        T = T[0] if len(T) > 0 else None
    if T is not None and hasattr(T, "position"):
        tgt = T.position
        return geodetic_to_enu(float(tgt.lat), float(tgt.lon), float(tgt.alt),
                               ref.lat, ref.lon, ref.alt)

    # 2) Use geodetics ONLY if they look valid (not zeros / NaNs)
    if all(k in c for k in ('lat', 'lon', 'alt')):
        lat, lon, alt = float(c['lat']), float(c['lon']), float(c['alt'])
        if not (abs(lat) < 1e-9 and abs(lon) < 1e-9 and abs(alt) < 1e-3):
            return geodetic_to_enu(lat, lon, alt, ref.lat, ref.lon, ref.alt)

    # 3) Last resort: rotate+translate local radar vector (already correct)
    src = c.get('source_pos', ref)
    p_src_in_ref = geodetic_to_enu(src.lat, src.lon, src.alt, ref.lat, ref.lon, ref.alt)
    from radar.tracking.helpers.enu_utils import enu_rotation
    v_local = np.array(to_cart(float(c['az']), float(c['el']), float(c['d'])), dtype=float)
    R = enu_rotation(src.lat, src.lon, ref.lat, ref.lon)
    return p_src_in_ref + R @ v_local


def get_track_id(c) -> int:
    """
    Extract track ID from cluster.

    For deception (ghosts): Creates distinct track IDs via position hash to enable
    multi-ghost clutter tracking, while preserving engagement_id for targeting.

    Priority:
    1. Real target ID from 'T' field (if present)
    2. Hash-based ID from (engagement_id, az, el, d) for ghost tracks
    3. Fallback to position hash for edge cases

    Args:
        c: Detection cluster

    Returns:
        Track ID for tracking (engagement_id preserved separately in track metadata)
    """
    # 1. Try to get real target ID from 'T' field first
    T = c.get('T', None)

    # Handle list-type T (legacy format from clustering)
    if isinstance(T, list) and len(T) == 0:
        T = None
    elif isinstance(T, list) and len(T) > 0:
        T = T[0]  # Take first target

    tid = getattr(T, 'id', None) if T is not None else None
    if tid is not None:
        # Real target with ground truth
        return tid

    # 2. For deception ghosts: derive track_id from (engagement_id, rounded position)
    # This creates multiple distinct tracks per jammer (dense clutter effect)
    engagement_id = c.get('engagement_id', None)
    if engagement_id is not None:
        # Hash includes engagement_id to namespace ghosts per jammer
        # Plus rounded angles/range to create spatial separation
        return hash((
            engagement_id,
            round(c.get('az', 0), 1),  # 0.1 deg resolution
            round(c.get('el', 0), 1),
            round(c.get('d', 0) / 100, 0)  # ~100m range bins
        )) % (2**31)

    # 3. Fallback: use hash of cluster position (should rarely happen)
    return hash((round(c.get('az', 0), 2), round(c.get('el', 0), 2), round(c.get('d', 0), 0))) % (2**31)



def associate_measurements(measurements: List[Tuple], default_ref: Position) -> List[Tuple]:
    """
    Simple 1:1 association passthrough.

    Input items are (track_id, cluster, measurement_enu, n_obs, used_ref).
    We return the same tuples so downstream code can use the correct per-measurement ref.
    The 'default_ref' parameter is kept for backward compatibility and is ignored.
    """
    assoc: List[Tuple] = []
    for tid, c, meas, n_obs, used_ref in measurements:
        assoc.append((tid, c, meas, n_obs, used_ref))
    return assoc