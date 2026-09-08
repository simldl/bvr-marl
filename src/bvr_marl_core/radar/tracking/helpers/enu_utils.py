import math

import numpy as np

_IDENTITY3 = np.eye(3)


def enu_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    Compute East-North-Up (ENU) basis vectors at given latitude/longitude.

    Args:
        lat_deg: Latitude in degrees
        lon_deg: Longitude in degrees

    Returns:
        3x3 matrix with ENU basis vectors as columns
    """
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # Columns are the ENU basis vectors [e, n, u]; orthonormal by construction,
    # no QR needed. (QR would introduce inconsistent sign flips.)
    return np.array(
        [
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0.0, cos_lat, sin_lat],
        ]
    )


def enu_rotation(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> np.ndarray:
    """
    Compute rotation matrix from one ENU frame to another.

    Maps vectors expressed in ENU(from) into ENU(to):
        v_to = R @ v_from,  with  R = B_to @ B_from^T

    Args:
        from_lat: Source latitude in degrees
        from_lon: Source longitude in degrees
        to_lat: Target latitude in degrees
        to_lon: Target longitude in degrees

    Returns:
        3x3 rotation matrix
    """
    if from_lat == to_lat and from_lon == to_lon:
        return _IDENTITY3.copy()

    B_from = enu_basis(from_lat, from_lon)
    B_to = enu_basis(to_lat, to_lon)
    # Both bases are orthonormal by construction, so their product is already a
    # proper rotation (det = +1) up to floating-point rounding; no SVD projection
    # is needed.
    return B_to.T @ B_from
