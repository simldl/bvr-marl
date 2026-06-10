"""
Geodesics computations
"""

import math

from geographiclib.geodesic import Geodesic

from bvr_marl_core.simulator.utils.angles import normalize_angle

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


def geodetic_direct(
    lat: float,
    lon: float,
    alt: float,
    yaw_deg: float,
    distance: float,
    vertical_distance: float = 0,
) -> tuple[float, float, float]:
    d = Geodesic.WGS84.Direct(
        lat, lon, yaw_deg, distance, outmask=Geodesic.LATITUDE | Geodesic.LONGITUDE
    )
    return d["lat2"], d["lon2"], alt + vertical_distance
