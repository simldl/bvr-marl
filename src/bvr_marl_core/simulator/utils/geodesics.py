"""
Geodesics computations
"""

import math

import numpy as np
from geographiclib.geodesic import Geodesic

from bvr_marl_core.simulator.utils.angles import normalize_angle

# WGS84 constants for geodetic<->ECEF<->ENU transforms.
_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563
_WGS84_B = _WGS84_A * (1 - _WGS84_F)
_WGS84_E_SQ = _WGS84_F * (2 - _WGS84_F)
_WGS84_B2_OVER_A2 = (_WGS84_B * _WGS84_B) / (_WGS84_A * _WGS84_A)  # == 1 - e_sq


def geodetic_to_ecef_scalar(lat: float, lon: float, alt: float) -> tuple[float, float, float]:
    """Convert geodetic lat/lon/alt (deg, m) to ECEF x/y/z (m)."""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E_SQ * sin_lat * sin_lat)
    x = (n + alt) * cos_lat * math.cos(lon_rad)
    y = (n + alt) * cos_lat * math.sin(lon_rad)
    z = (_WGS84_B2_OVER_A2 * n + alt) * sin_lat
    return x, y, z


def geodetic_to_enu(
    lat: float, lon: float, alt: float, ref_lat: float, ref_lon: float, ref_alt: float
) -> np.ndarray:
    """Convert geodetic lat/lon/alt to local ENU (east, north, up) about a reference."""
    x, y, z = geodetic_to_ecef_scalar(lat, lon, alt)
    xr, yr, zr = geodetic_to_ecef_scalar(ref_lat, ref_lon, ref_alt)
    dx, dy, dz = x - xr, y - yr, z - zr

    lat0 = math.radians(ref_lat)
    lon0 = math.radians(ref_lon)
    sin_lat0, cos_lat0 = math.sin(lat0), math.cos(lat0)
    sin_lon0, cos_lon0 = math.sin(lon0), math.cos(lon0)

    e = -sin_lon0 * dx + cos_lon0 * dy
    n = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
    u = cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz
    return np.array((e, n, u))


# Flat-earth bearing formula (midpoint cosine approximation) is accurate to:
#   100 km → <0.05° bearing error,  150 km → <0.1° error — well within DLZ/SQI tolerances.
# Using 1.5° covers pairs up to ~165 km at the equator (or ~110 km at lat 48°), which
# encompasses most BVR combat ranges and eliminates the majority of Geodesic.WGS84.Inverse calls.
_FAST_PATH_THRESHOLD_DEG = 1.5  # ~165 km at equator

_bearing_cache = {}
_CACHE_PRECISION = 6
_MAX_CACHE_SIZE = 10000


def geodetic_distance_km(
    lat_1: float, lon_1: float, alt_1: float, lat_2: float, lon_2: float, alt_2: float
) -> float:
    """
    Compute geodetic distance. Uses flat-earth approximation when points are close,
    falling back to accurate geodesic calculation for longer distances.
    """
    lat_diff = abs(lat_2 - lat_1)
    lon_diff = abs(lon_2 - lon_1)

    if lat_diff < _FAST_PATH_THRESHOLD_DEG and lon_diff < _FAST_PATH_THRESHOLD_DEG:
        lat_avg = (lat_1 + lat_2) / 2.0
        meters_per_deg_lat = 111132.92
        meters_per_deg_lon = 111132.92 * math.cos(math.radians(lat_avg))

        dx = lon_diff * meters_per_deg_lon
        dy = lat_diff * meters_per_deg_lat
        dz = abs(alt_2 - alt_1)

        horizontal_distance = math.sqrt(dx**2 + dy**2) / 1000.0
        vertical_distance = dz / 1000.0
        return math.sqrt(horizontal_distance**2 + vertical_distance**2)

    r = Geodesic.WGS84.Inverse(lat_1, lon_1, lat_2, lon_2, outmask=Geodesic.DISTANCE)
    horizontal_distance = r["s12"] / 1000.0
    vertical_distance = abs(alt_2 - alt_1) / 1000.0
    return (horizontal_distance**2 + vertical_distance**2) ** 0.5


def geodetic_bearing_deg(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    """
    Compute geodetic bearing. Uses flat-earth approximation when points are close,
    with per-tick caching to avoid redundant calculations.
    """
    cache_key = (
        round(lat_1, _CACHE_PRECISION),
        round(lon_1, _CACHE_PRECISION),
        round(lat_2, _CACHE_PRECISION),
        round(lon_2, _CACHE_PRECISION),
    )

    if cache_key in _bearing_cache:
        return _bearing_cache[cache_key]

    lat_diff = abs(lat_2 - lat_1)
    lon_diff = abs(lon_2 - lon_1)

    if lat_diff < _FAST_PATH_THRESHOLD_DEG and lon_diff < _FAST_PATH_THRESHOLD_DEG:
        lat_avg = (lat_1 + lat_2) / 2.0
        x = math.cos(math.radians(lat_avg)) * (lon_2 - lon_1)
        y = lat_2 - lat_1
        bearing = math.degrees(math.atan2(x, y))
        result = normalize_angle(bearing)
    else:
        r = Geodesic.WGS84.Inverse(lat_1, lon_1, lat_2, lon_2, outmask=Geodesic.AZIMUTH)
        result = normalize_angle(r["azi1"])

    if len(_bearing_cache) < _MAX_CACHE_SIZE:
        _bearing_cache[cache_key] = result

    return result


def clear_bearing_cache():
    """Clear the bearing cache. Should be called at the start of each simulation step."""
    global _bearing_cache
    _bearing_cache = {}


# WGS84 constants for the local-curvature fast path in geodetic_direct.
_WGS84_A = 6378137.0
_WGS84_E_SQ = (1 / 298.257223563) * (2 - 1 / 298.257223563)

# For steps below this length the local-radii step formula differs from the exact
# geodesic by at most a few centimeters (error grows ~quadratically with step
# length); per-tick movement steps are a few hundred meters.
_DIRECT_FAST_PATH_M = 2000.0


def geodetic_direct(
    lat: float,
    lon: float,
    alt: float,
    yaw_deg: float,
    distance: float,
    vertical_distance: float = 0,
) -> tuple[float, float, float]:
    if abs(distance) < _DIRECT_FAST_PATH_M:
        # Local-curvature step: meridian radius M for latitude, prime-vertical
        # radius N for longitude. Centimeter-scale error at these step lengths.
        lat_rad = math.radians(lat)
        sin_lat = math.sin(lat_rad)
        w_sq = 1.0 - _WGS84_E_SQ * sin_lat * sin_lat
        w = math.sqrt(w_sq)
        m_radius = _WGS84_A * (1.0 - _WGS84_E_SQ) / (w_sq * w)  # meridian radius
        n_radius = _WGS84_A / w  # prime vertical radius

        yaw_rad = math.radians(yaw_deg)
        lat2 = lat + math.degrees(distance * math.cos(yaw_rad) / m_radius)
        lon2 = lon + math.degrees(distance * math.sin(yaw_rad) / (n_radius * math.cos(lat_rad)))
        return lat2, lon2, alt + vertical_distance

    d = Geodesic.WGS84.Direct(
        lat, lon, yaw_deg, distance, outmask=Geodesic.LATITUDE | Geodesic.LONGITUDE
    )
    return d["lat2"], d["lon2"], alt + vertical_distance
