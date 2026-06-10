import hashlib
import struct
from typing import Any

import numpy as np

from bvr_marl_core.radar.core.utils import geodetic_to_enu, to_cart
from bvr_marl_core.simulator.core.helpers import Position


def _stable_hash(*parts) -> int:
    """
    Deterministic 31-bit hash that is stable across Python processes.
    Encodes each part as a UTF-8 string, hashes with SHA-1, and returns
    the lower 31 bits of the digest as a non-negative integer.
    """
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return struct.unpack(">I", h.digest()[:4])[0] & 0x7FFF_FFFF


def build_measurements_with_ref(clusters, track_refs: dict, default_ref: Position) -> list[tuple]:
    """
    Build measurements from clusters, determining ENU reference for each track.

    Returns tuples: (track_id, cluster, measurement_enu, n_obs, used_ref)
    """
    out: list[tuple[Any, dict, np.ndarray, int, Position]] = []
    for c in clusters:
        tid = get_track_id(c)

        ref = track_refs.get(tid)
        if ref is None:
            ref = c.get("source_pos", None) or default_ref
            track_refs[tid] = ref.copy()

        meas_enu = cluster_to_track_measurement(c, ref)
        n_obs = int(c.get("n_obs", 1))
        out.append((tid, c, meas_enu, n_obs, ref.copy()))
    return out


def cluster_to_track_measurement(c, ref: Position) -> np.ndarray:
    override = c.get("measurement_position", None)
    if override is not None:
        return geodetic_to_enu(
            float(override.lat),
            float(override.lon),
            float(override.alt),
            ref.lat,
            ref.lon,
            ref.alt,
        )

    # Prefer datalink (GT); c['T'] can be a list — unwrap it.
    T = c.get("T", None)
    if isinstance(T, list):
        T = T[0] if len(T) > 0 else None
    if T is not None and hasattr(T, "position"):
        tgt = T.position
        return geodetic_to_enu(
            float(tgt.lat), float(tgt.lon), float(tgt.alt), ref.lat, ref.lon, ref.alt
        )

    # Use geodetics if they look valid (not zeros / NaNs)
    if all(k in c for k in ("lat", "lon", "alt")):
        lat, lon, alt = float(c["lat"]), float(c["lon"]), float(c["alt"])
        if not (abs(lat) < 1e-9 and abs(lon) < 1e-9 and abs(alt) < 1e-3):
            return geodetic_to_enu(lat, lon, alt, ref.lat, ref.lon, ref.alt)

    # Last resort: rotate+translate local radar vector
    src = c.get("source_pos", ref)
    p_src_in_ref = geodetic_to_enu(src.lat, src.lon, src.alt, ref.lat, ref.lon, ref.alt)
    from bvr_marl_core.radar.tracking.helpers.enu_utils import enu_rotation

    v_local = np.array(to_cart(float(c["az"]), float(c["el"]), float(c["d"])), dtype=float)
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
    T = c.get("T", None)

    if isinstance(T, list) and len(T) == 0:
        T = None
    elif isinstance(T, list) and len(T) > 0:
        T = T[0]

    tid = getattr(T, "id", None) if T is not None else None
    if tid is not None:
        return tid

    # For deception ghosts: derive track_id from (engagement_id, rounded position)
    # to create multiple distinct tracks per jammer (dense clutter effect).
    engagement_id = c.get("engagement_id", None)
    if engagement_id is not None:
        return _stable_hash(
            engagement_id,
            round(c.get("az", 0), 1),  # 0.1 deg resolution
            round(c.get("el", 0), 1),
            round(c.get("d", 0) / 100, 0),  # ~100 m range bins
        )

    # Fallback: deterministic hash of cluster position
    return _stable_hash(
        round(c.get("az", 0), 2),
        round(c.get("el", 0), 2),
        round(c.get("d", 0), 0),
    )


def associate_measurements(measurements: list[tuple], default_ref: Position) -> list[tuple]:
    """
    Simple 1:1 association passthrough.

    Input items are (track_id, cluster, measurement_enu, n_obs, used_ref).
    We return the same tuples so downstream code can use the correct per-measurement ref.
    The 'default_ref' parameter is kept for backward compatibility and is ignored.
    """
    assoc: list[tuple] = []
    for tid, c, meas, n_obs, used_ref in measurements:
        assoc.append((tid, c, meas, n_obs, used_ref))
    return assoc
