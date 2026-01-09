import math
import numpy as np


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
    
    # ENU basis vectors
    e = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=float)
    n = np.array([-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat)], dtype=float)
    u = np.array([math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat)], dtype=float)
    
    B = np.column_stack([e, n, u])
    
    # QR decomposition for numerical stability
    Q, _ = np.linalg.qr(B)
    return Q


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
    B_from = enu_basis(from_lat, from_lon)
    B_to = enu_basis(to_lat, to_lon)
    R = B_to.T @ B_from
    
    # Project to nearest rotation matrix using SVD (numerical safety)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    
    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
        
    return R