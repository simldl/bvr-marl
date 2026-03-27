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
    ct_filter = CoordinatedTurnKFFilter(initial_state_8d, cov_8x8, Q_8x8, R_3x3, dt=0.1)
    imm = IMMFilter([cv_filter, ct_filter], transition_matrix, [0.8, 0.2], measurement_cov)
"""

# Import all filter classes for easy access
from air_to_air_rl.radar.tracking.filter.base_filter import BaseKFFilter
from air_to_air_rl.radar.tracking.filter.constant_velocity_filter import ConstantVelocityKFFilter
from air_to_air_rl.radar.tracking.filter.coordinated_turn_filter import CoordinatedTurnKFFilter
from air_to_air_rl.radar.tracking.filter.imm_filter import IMMFilter

# Public API
__all__ = [
    "BaseKFFilter",
    "ConstantVelocityKFFilter",
    "CoordinatedTurnKFFilter",
    "IMMFilter",
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


def create_imm_cv_ct_filter(initial_position, dt=0.1, cv_weight=0.8):
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
    cv_filter = create_cv_filter(initial_position, dt)

    # Create CT filter (simplified initialization)
    initial_state_8d = np.array(
        [
            initial_position[0],
            initial_position[1],
            initial_position[2],  # position
            100.0,
            0.0,
            0.0,  # velocity magnitude, yaw, pitch
            0.0,
            0.0,  # turn rates
        ]
    )
    ct_filter = CoordinatedTurnKFFilter(
        initial_state=initial_state_8d,
        initial_covariance=np.diag([100, 100, 40, 400, 180, 90, 30, 30]),
        process_noise_cov=np.diag([1, 1, 1, 100, 10, 10, 5, 5]),
        measurement_noise_cov=np.eye(3) * 1.0,
        dt=dt,
    )

    # IMM configuration
    transition_matrix = np.array([[0.95, 0.05], [0.05, 0.95]])
    mode_probs = [cv_weight, 1.0 - cv_weight]
    measurement_cov = np.eye(3) * 1.0

    return IMMFilter([cv_filter, ct_filter], transition_matrix, mode_probs, measurement_cov)
