import math

import numpy as np

from air_to_air_rl.simulator.core.helpers import Position
from air_to_air_rl.simulator.core.units import Unit
from air_to_air_rl.simulator.utils.angles import signed_yaw_deg_diff, yaw_geo_to_math
from air_to_air_rl.simulator.utils.geodesics import geodetic_bearing_deg

rad_earth = 6378137.0


def geodetic_to_enu(
    lat: float, lon: float, alt: float, ref_lat: float, ref_lon: float, ref_alt: float
) -> np.ndarray:

    # WGS84
    a = rad_earth
    f = 1 / 298.257223563
    b = a * (1 - f)
    e_sq = f * (2 - f)

    def geodetic_to_ecef(lat, lon, alt):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        N = a / np.sqrt(1 - e_sq * np.sin(lat_rad) ** 2)
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = ((b**2 / a**2) * N + alt) * np.sin(lat_rad)
        return np.array([x, y, z])

    xyz = geodetic_to_ecef(lat, lon, alt)
    xyz_ref = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)

    dx = xyz - xyz_ref

    lat0 = np.radians(ref_lat)
    lon0 = np.radians(ref_lon)
    R = np.array(
        [
            [-np.sin(lon0), np.cos(lon0), 0],
            [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
            [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
        ]
    )

    enu = R @ dx
    return enu


def enu_to_geodetic(enu_vec, ref_lat, ref_lon, ref_alt):
    # WGS84
    a = 6378137.0
    f = 1 / 298.257223563
    e_sq = f * (2 - f)

    def geodetic_to_ecef(lat, lon, alt):
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)
        N = a / np.sqrt(1 - e_sq * np.sin(lat_r) ** 2)
        x = (N + alt) * np.cos(lat_r) * np.cos(lon_r)
        y = (N + alt) * np.cos(lat_r) * np.sin(lon_r)
        z = (N * (1 - e_sq) + alt) * np.sin(lat_r)
        return np.array([x, y, z])

    def ecef_to_geodetic(x, y, z):
        # Bowring’s method (robust & fast)
        b = a * (1 - f)
        ep_sq = (a * a - b * b) / (b * b)
        p = np.sqrt(x * x + y * y)
        th = np.arctan2(a * z, b * p)
        lon = np.arctan2(y, x)
        lat = np.arctan2(z + ep_sq * b * np.sin(th) ** 3, p - e_sq * a * np.cos(th) ** 3)
        N = a / np.sqrt(1 - e_sq * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - N
        return np.degrees(lat), np.degrees(lon), float(alt)

    ref_xyz = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)
    lat0 = np.radians(ref_lat)
    lon0 = np.radians(ref_lon)
    R = np.array(
        [
            [-np.sin(lon0), np.cos(lon0), 0.0],
            [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
            [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
        ]
    )
    ecef = ref_xyz + R.T @ np.asarray(enu_vec, dtype=float)
    return ecef_to_geodetic(*ecef)


def transfer_enu_between_refs(
    enu: np.ndarray, old_ref: "Position", new_ref: "Position"
) -> "Position":

    lat, lon, alt = enu_to_geodetic(enu, old_ref.lat, old_ref.lon, old_ref.alt)
    enu_new = geodetic_to_enu(lat, lon, alt, new_ref.lat, new_ref.lon, new_ref.alt)
    lat_new, lon_new, alt_new = enu_to_geodetic(enu_new, new_ref.lat, new_ref.lon, new_ref.alt)
    return Position(lat_new, lon_new, alt_new)


def _effective_rcs(
    target: Unit, rp: Position, _dE_r2t: float = None, _dN_r2t: float = None, _dU_r2t: float = None
) -> float:
    """
    Effective monostatic RCS with azimuth AND elevation dependence.

    σ_eff = σ0 * f_az(φ) * f_el(ϑ)

    - φ  : azimuth aspect (deg) between target yaw and LOS bearing (0° nose-on, 90° beam, 180° tail-on)
    - ϑ  : elevation aspect (deg) of the LOS relative to local horizontal (positive = looking from above),
           optionally corrected by target pitch if available.

    Optional pre-computed ENU deltas (radar→target, from _angles_dist):
        _dE_r2t, _dN_r2t, _dU_r2t — if provided, skip redundant geodetic_bearing and ENU recomputation.

    Per-aircraft tuning (optional): target.rcs_pattern = {
        "front_floor": 0.30,   # min azimuth multiplier at φ=0° (nose-on), 0..1
        "tail_floor":  0.50,   # min azimuth multiplier at φ=180° (tail-on), 0..1
        "n_az":        1.2,    # azimuth shaping exponent (beam peak sharpness)
        "k_top":       0.20,   # elevation gain when seen from above  (multiplier ~ 1 + k_top * sin^n_el)
        "k_bottom":    0.35,   # elevation gain when seen from below
        "n_el":        1.5,    # elevation shaping exponent
        "sigma_min":   0.05,   # lower clamp as fraction of σ0
        "sigma_max":   3.00,   # upper clamp as multiple of σ0
    }
    """
    base_rcs = float(getattr(target, "rcs", 1.0))

    if not hasattr(target, "position") or rp is None:
        return base_rcs

    # LOS geometry: bearing & elevation (absolute geodetic)
    if _dE_r2t is not None:
        # Fast path: use pre-computed ENU deltas from _angles_dist (radar→target direction).
        # Reverse to get target→radar direction for aspect angle computation.
        dE = -_dE_r2t  # target→radar East
        dN = -_dN_r2t  # target→radar North
        dU = -_dU_r2t  # target→radar Up
        # Bearing from target to radar using flat-earth atan2 (matches _angles_dist convention)
        bear = math.degrees(math.atan2(dE, dN))
        if bear < 0.0:
            bear += 360.0
    else:
        # Full geodetic bearing computation (fallback when ENU not available)
        bear = geodetic_bearing_deg(target.position.lat, target.position.lon, rp.lat, rp.lon)
        dN = (rp.lat - target.position.lat) * 111_000.0
        dE = (rp.lon - target.position.lon) * 111_000.0 * math.cos(math.radians(rp.lat))
        dU = rp.alt - target.position.alt

    yaw_geo = float(getattr(target, "yaw_deg", getattr(target, "orientation", 0.0)))
    phi = abs(signed_yaw_deg_diff(yaw_geo, bear))  # deg in [0, 180]

    # Elevation aspect: atan2(ΔU, sqrt(ΔE²+ΔN²)); corrected by target pitch
    hd = math.hypot(dE, dN)
    el_abs = math.degrees(math.atan2(dU, max(hd, 1e-6)))
    pitch = float(
        getattr(target, "pitch_deg", getattr(getattr(target, "control", None), "pitch_deg", 0.0))
    )
    el_body = el_abs - pitch  # relative to body pitch (no roll available)

    pat = getattr(target, "rcs_pattern", {}) or {}
    front_floor = float(pat.get("front_floor", 0.30))  # φ=0°
    tail_floor = float(pat.get("tail_floor", 0.50))  # φ=180°
    n_az = max(0.5, float(pat.get("n_az", 1.2)))  # beam shaping
    k_top = float(pat.get("k_top", 0.20))
    k_bottom = float(pat.get("k_bottom", 0.35))
    n_el = max(0.5, float(pat.get("n_el", 1.5)))
    sigma_min_f = max(0.0, float(pat.get("sigma_min", 0.05)))
    sigma_max_f = max(1.0, float(pat.get("sigma_max", 3.00)))

    # Azimuth multiplier f_az(φ): beam peak at 90°, floors at nose (0°) and tail (180°)
    s = math.sin(math.radians(min(180.0, max(0.0, phi)))) ** n_az  # 0..1
    w_front = max(0.0, 1.0 - phi / 180.0)
    w_tail = 1.0 - w_front
    f_az = (w_front * front_floor) + (w_tail * tail_floor) + (1.0 - w_front - w_tail) * 1.0
    f_az = max(
        min(f_az + (1.0 - max(front_floor, tail_floor)) * s, 1.0), min(front_floor, tail_floor)
    )

    # Elevation multiplier f_el(ϑ): dorsal (above) gains k_top, ventral (below) gains k_bottom
    sev = abs(math.sin(math.radians(el_body))) ** n_el
    if el_body >= 0.0:
        f_el = 1.0 + k_top * sev
    else:
        f_el = 1.0 + k_bottom * sev

    sigma = base_rcs * f_az * f_el
    sigma = max(sigma_min_f * base_rcs, min(sigma_max_f * base_rcs, sigma))
    return float(sigma)


def _angles_dist(rp: Position, h: float, p: float, tp: Position) -> tuple[float, float, float]:
    """
    Compute *relative* angles from the radar boresight to the target, plus slant range.

    Returns:
        az_rel_deg : signed azimuth *relative to yaw h*  (deg, +right/-left)
        el_rel_deg : signed elevation *relative to pitch p* (deg, +up/-down)
        dist_m     : 3D slant range (m)

    Notes:
        - ENU frame (E,N,U)
        - Geodetic azimuth (0°=North, 90°=East, clockwise positive)
        - We return RELATIVE angles for FOV gating. For absolute angles use:
              az_abs = h + az_rel,  el_abs = p + el_rel
    """
    dN = (tp.lat - rp.lat) * 111_000.0
    dE = (tp.lon - rp.lon) * 111_000.0 * math.cos(math.radians(rp.lat))
    dU = tp.alt - rp.alt

    hd = math.hypot(dE, dN)
    dist = math.hypot(hd, dU)

    az_abs = math.degrees(math.atan2(dE, dN))  # 0°=N, +CW to E
    el_abs = math.degrees(math.atan2(dU, max(hd, 1e-12)))  # +up
    az_rel = signed_yaw_deg_diff(h, az_abs)  # signed, for FOV gate
    el_rel = el_abs - p

    return az_rel, el_rel, dist


def _doppler(target: Unit, az_abs_deg: float, el_abs_deg: float, freq_hz: float) -> float:
    """
    Doppler shift: f_d = (2 * f / c) * (v ⋅ r_hat)
    Inputs:
        az_abs_deg : absolute geodetic azimuth (0°=N, 90°=E, +CW)
        el_abs_deg : absolute elevation (+up)
    Assumes target.velocity is ENU: (E, N, U).
    """
    if not hasattr(target, "velocity"):
        return 0.0
    ar = math.radians(az_abs_deg)
    er = math.radians(el_abs_deg)
    # ENU unit LOS for geodetic azimuth
    r_hat = np.array(
        [
            math.cos(er) * math.sin(ar),  # E
            math.cos(er) * math.cos(ar),  # N
            math.sin(er),
        ]
    )  # U
    vrel = float(np.dot(np.asarray(target.velocity, dtype=float), r_hat))
    return 2.0 * freq_hz * vrel / 299_792_458.0


def to_cart(az_abs_deg: float, el_abs_deg: float, dist_m: float) -> tuple[float, float, float]:
    """
    Spherical (az, el, R) → ENU Cartesian (E, N, U) with *geodetic* azimuth:
      - az: 0°=North, 90°=East, clockwise positive
      - el: elevation from local horizontal, +up
    """
    ar = math.radians(az_abs_deg)
    er = math.radians(el_abs_deg)
    E = dist_m * math.cos(er) * math.sin(ar)
    N = dist_m * math.cos(er) * math.cos(ar)
    U = dist_m * math.sin(er)
    return (E, N, U)


def measurement_to_cartesian(detection: dict) -> np.ndarray:
    """Convert a single detection dictionary to Cartesian coordinates."""
    return np.array(to_cart(detection["az"], detection["el"], detection["d"]))
