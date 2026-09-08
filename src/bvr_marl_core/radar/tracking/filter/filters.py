"""
Kalman Filter implementations for target tracking.

This module provides a clean, modular interface to different filter types:
- BaseKFFilter: Abstract interface for all filters
- ConstantVelocityKFFilter: Linear Kalman filter for constant velocity motion
- CoordinatedTurnKFFilter: Extended Kalman filter for coordinated turn motion
- IMMFilter: Interacting Multiple Model filter combining multiple filter types

All filters operate in ENU (East-North-Up) coordinates and provide a consistent
interface with predict(), update(), get_state(), and get_velocity() methods.

Example:
    # Basic constant velocity tracking
    filter = ConstantVelocityKFFilter(initial_pos, dt=0.1)
    filter.predict(dt)
    filter.update(measurement)
    position, velocity = filter.get_state()[:3], filter.get_state()[3:]

    # Multiple model tracking
    cv_filter = ConstantVelocityKFFilter(initial_pos, dt=0.1)
    ct_filter = CoordinatedTurnKFFilter(initial_state_7d, cov_7x7, Q_7x7, R_3x3, dt=0.1)
    imm = IMMFilter([cv_filter, ct_filter], transition_matrix, [0.8, 0.2], measurement_cov)
"""

# Import all filter classes for easy access
from bvr_marl_core.radar.tracking.filter.base_filter import BaseKFFilter
from bvr_marl_core.radar.tracking.filter.constant_velocity_filter import ConstantVelocityKFFilter
from bvr_marl_core.radar.tracking.filter.coordinated_turn_filter import CoordinatedTurnKFFilter
from bvr_marl_core.radar.tracking.filter.imm_filter import IMMFilter

# Public API
__all__ = [
    "BaseKFFilter",
    "ConstantVelocityKFFilter",
    "CoordinatedTurnKFFilter",
    "IMMFilter",
    "create_ct_filter",
    "create_imm_cv_ct_filter",
]


def create_cv_filter(initial_position, dt=0.1, process_noise=20.0, measurement_noise=1.0):
    """
    Convenience function to create a constant velocity filter.

    Args:
        initial_position: Initial 3D position measurement [x, y, z]
        dt: Time step in seconds
        process_noise: Process noise variance
        measurement_noise: Measurement noise variance

    Returns:
        Configured ConstantVelocityKFFilter instance
    """
    return ConstantVelocityKFFilter(
        initial_measurement=initial_position,
        dt=dt,
        process_noise_std=process_noise,
        measurement_noise_var=measurement_noise,
    )


def create_ct_filter(initial_position, dt=0.1, measurement_covariance=None):
    """Create a Cartesian CT filter with synthetic tracker-scale tuning."""
    import numpy as np

    measurement_covariance = (
        np.eye(3)
        if measurement_covariance is None
        else np.asarray(measurement_covariance, dtype=float)
    )
    initial_state = np.concatenate(
        [np.asarray(initial_position, dtype=float), np.zeros(4, dtype=float)]
    )
    initial_covariance = np.zeros((7, 7), dtype=float)
    initial_covariance[:3, :3] = measurement_covariance
    initial_covariance[3:6, 3:6] = np.eye(3) * 90_000.0
    initial_covariance[6, 6] = np.radians(3.0) ** 2
    return CoordinatedTurnKFFilter(
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        process_noise_cov=np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, np.radians(0.05) ** 2]),
        measurement_noise_cov=measurement_covariance,
        dt=dt,
    )


def create_imm_cv_ct_filter(
    initial_position,
    dt=0.1,
    cv_weight=0.8,
    measurement_covariance=None,
):
    """
    Convenience function to create a CV-CT IMM filter.

    Args:
        initial_position: Initial 3D position measurement [x, y, z]
        dt: Time step in seconds
        cv_weight: Weight for constant velocity model (0.0 to 1.0)

    Returns:
        Configured IMMFilter with CV and CT models
    """
    import numpy as np

    # Create CV filter
    measurement_covariance = (
        np.eye(3)
        if measurement_covariance is None
        else np.asarray(measurement_covariance, dtype=float)
    )
    cv_filter = ConstantVelocityKFFilter(
        initial_position,
        dt,
        process_noise_std=8.0,
        R_diag=tuple(np.diag(measurement_covariance)),
    )

    ct_filter = create_ct_filter(initial_position, dt, measurement_covariance)

    # IMM configuration
    transition_matrix = np.array([[0.97, 0.03], [0.03, 0.97]])
    mode_probs = [cv_weight, 1.0 - cv_weight]
    return IMMFilter([cv_filter, ct_filter], transition_matrix, mode_probs, measurement_covariance)
