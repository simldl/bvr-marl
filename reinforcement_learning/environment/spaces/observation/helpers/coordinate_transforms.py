"""
Coordinate transformation utilities.
Handles ENU (East-North-Up) coordinate systems and velocity components.
"""
import numpy as np
import math


def enu_delta_meters(unit, other):
    """
    Local small-angle ENU delta from unit -> other (meters).
    E = Δlon * cos(lat) * 111e3,  N = Δlat * 111e3,  U = Δalt

    Args:
        unit: Reference unit
        other: Target unit

    Returns:
        (dE, dN, dU): East, North, Up deltas in meters
    """
    dE = (other.position.lon - unit.position.lon) * 111_000.0 * math.cos(math.radians(unit.position.lat))
    dN = (other.position.lat - unit.position.lat) * 111_000.0
    dU = other.position.alt - unit.position.alt
    return dE, dN, dU


def velocity_components(speed, pitch_deg, yaw_deg):
    """
    Geographic yaw convention: 0°=North, 90°=East.
    East = Vh * sin(yaw),  North = Vh * cos(yaw),  Up = V * sin(pitch).

    Args:
        speed: Magnitude of velocity (m/s)
        pitch_deg: Pitch angle (degrees)
        yaw_deg: Yaw angle (degrees)

    Returns:
        (vx_E, vy_N, vz_U): Velocity components in ENU frame
    """
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    Vh = speed * math.cos(p)
    vx_E = Vh * math.sin(y)  # East
    vy_N = Vh * math.cos(y)  # North
    vz_U = speed * math.sin(p)
    return vx_E, vy_N, vz_U


def rel_state(unit, other):
    """
    Relative state from unit -> other in ENU:
    [dE, dN, dU, dvE, dvN, dvU].

    Args:
        unit: Reference unit
        other: Target unit

    Returns:
        np.ndarray: 6-element array [dE, dN, dU, dvE, dvN, dvU]
    """
    dE, dN, dU = enu_delta_meters(unit, other)

    so = float(getattr(other, "speed", 0.0))
    po = float(getattr(other, "pitch_deg", 0.0))
    yo = float(getattr(other, "yaw_deg", 0.0))
    vx_o, vy_o, vz_o = velocity_components(so, po, yo)

    su = float(getattr(unit, "speed", 0.0))
    pu = float(getattr(unit, "pitch_deg", 0.0))
    yu = float(getattr(unit, "yaw_deg", 0.0))
    vx_u, vy_u, vz_u = velocity_components(su, pu, yu)

    return np.array([dE, dN, dU, vx_o - vx_u, vy_o - vy_u, vz_o - vz_u], dtype=np.float32)


def rel_position(unit, other):
    """Relative ENU position from unit -> other: [dE, dN, dU] (meters)."""
    return np.array(enu_delta_meters(unit, other), dtype=np.float32)


def rel_velocity(unit, other):
    """Relative ENU velocity (other − unit): [dvE, dvN, dvU] (m/s)."""
    vx_o, vy_o, vz_o = velocity_components(
        float(getattr(other, "speed", 0.0)),
        float(getattr(other, "pitch_deg", 0.0)),
        float(getattr(other, "yaw_deg", 0.0)),
    )
    vx_u, vy_u, vz_u = velocity_components(
        float(getattr(unit, "speed", 0.0)),
        float(getattr(unit, "pitch_deg", 0.0)),
        float(getattr(unit, "yaw_deg", 0.0)),
    )
    return np.array([vx_o - vx_u, vy_o - vy_u, vz_o - vz_u], dtype=np.float32)
