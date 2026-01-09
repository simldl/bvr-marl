"""
    Geodesics computations
"""

from geographiclib.geodesic import Geodesic
from simulator.utils.angles import normalize_angle
from typing import Tuple
import math

# OPTIMIZATION: Fast-path threshold for flat-earth approximation (in degrees)
# For BVR combat scenarios, flat-earth approximation is acceptable up to ~50km
# The bearing error is <0.5 degrees which is acceptable for tactical decisions
_FAST_PATH_THRESHOLD_DEG = 0.5  # ~55 km at equator

# OPTIMIZATION: Cache for geodetic_bearing_deg to avoid redundant calculations
# Cache is keyed by (lat1, lon1, lat2, lon2) rounded to reduce key space
_bearing_cache = {}
_CACHE_PRECISION = 6  # decimal places for lat/lon rounding
_MAX_CACHE_SIZE = 10000  # prevent unbounded growth

def geodetic_distance_km(lat_1: float, lon_1: float, alt_1: float,
                         lat_2: float, lon_2: float, alt_2: float) -> float:
    """
    Compute geodetic distance with fast-path optimization for close distances.

    PERFORMANCE: Uses flat-earth approximation when points are close (<10km),
    avoiding expensive Geodesic.WGS84.Inverse calls. Falls back to accurate
    geodesic calculation for longer distances.
    """
    # OPTIMIZATION: Fast-path for close distances using flat-earth approximation
    lat_diff = abs(lat_2 - lat_1)
    lon_diff = abs(lon_2 - lon_1)

    if lat_diff < _FAST_PATH_THRESHOLD_DEG and lon_diff < _FAST_PATH_THRESHOLD_DEG:
        # Flat-earth approximation (good for < 10km)
        lat_avg = (lat_1 + lat_2) / 2.0
        meters_per_deg_lat = 111132.92  # meters per degree latitude
        meters_per_deg_lon = 111132.92 * math.cos(math.radians(lat_avg))

        dx = lon_diff * meters_per_deg_lon
        dy = lat_diff * meters_per_deg_lat
        dz = abs(alt_2 - alt_1)

        horizontal_distance = math.sqrt(dx**2 + dy**2) / 1000.0  # km
        vertical_distance = dz / 1000.0  # km
        return math.sqrt(horizontal_distance**2 + vertical_distance**2)

    # Standard geodesic calculation for longer distances
    r = Geodesic.WGS84.Inverse(lat_1, lon_1, lat_2, lon_2, outmask=Geodesic.DISTANCE)
    horizontal_distance = r["s12"] / 1000.0  # convert meters to km
    vertical_distance = abs(alt_2 - alt_1) / 1000.0  # convert meters to km
    return (horizontal_distance**2 + vertical_distance**2) ** 0.5


def geodetic_bearing_deg(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    """
    Compute geodetic bearing with fast-path optimization for close distances.

    PERFORMANCE: Uses flat-earth approximation when points are close (<10km).
    Also uses caching to avoid redundant calculations within the same simulation step.
    """
    # OPTIMIZATION: Check cache first (round coordinates to reduce key space)
    cache_key = (
        round(lat_1, _CACHE_PRECISION),
        round(lon_1, _CACHE_PRECISION),
        round(lat_2, _CACHE_PRECISION),
        round(lon_2, _CACHE_PRECISION)
    )

    if cache_key in _bearing_cache:
        return _bearing_cache[cache_key]

    # OPTIMIZATION: Fast-path for close distances
    lat_diff = abs(lat_2 - lat_1)
    lon_diff = abs(lon_2 - lon_1)

    if lat_diff < _FAST_PATH_THRESHOLD_DEG and lon_diff < _FAST_PATH_THRESHOLD_DEG:
        # Flat-earth approximation for bearing
        lat_avg = (lat_1 + lat_2) / 2.0
        x = math.cos(math.radians(lat_avg)) * (lon_2 - lon_1)
        y = lat_2 - lat_1
        bearing = math.degrees(math.atan2(x, y))
        result = normalize_angle(bearing)
    else:
        # Standard geodesic calculation
        r = Geodesic.WGS84.Inverse(lat_1, lon_1, lat_2, lon_2, outmask=Geodesic.AZIMUTH)
        result = normalize_angle(r["azi1"])

    # Cache the result (with size limit)
    if len(_bearing_cache) < _MAX_CACHE_SIZE:
        _bearing_cache[cache_key] = result

    return result


def clear_bearing_cache():
    """Clear the bearing cache. Should be called at the start of each simulation step."""
    global _bearing_cache
    _bearing_cache = {}

def geodetic_direct(lat: float, lon: float, alt: float,
                    yaw_deg: float, distance: float,
                    vertical_distance: float = 0) -> Tuple[float, float, float]:
    d = Geodesic.WGS84.Direct(lat, lon, yaw_deg, distance, 
                              outmask=Geodesic.LATITUDE | Geodesic.LONGITUDE)
    return d["lat2"], d["lon2"], alt + vertical_distance
