import numpy as np
import pytest

from bvr_marl_core.radar.tracking.filter.constant_velocity_filter import (
    ConstantVelocityKFFilter,
)


def test_continuous_white_acceleration_covariance_scales_with_dt():
    amplitude = 4.0
    filter_ = ConstantVelocityKFFilter(np.zeros(3), dt=2.0, process_noise_std=amplitude)

    q = amplitude**2
    assert filter_.process_noise_spectral_amplitude == amplitude
    assert filter_.Q[0, 0] == pytest.approx(q * 2.0**3 / 3.0)
    assert filter_.Q[0, 3] == pytest.approx(q * 2.0**2 / 2.0)
    assert filter_.Q[3, 3] == pytest.approx(q * 2.0)


def test_process_noise_parameter_is_compatible_but_not_discrete_acceleration():
    filter_ = ConstantVelocityKFFilter(np.zeros(3), dt=0.5, process_noise_std=3.0)

    assert filter_.sigma_a == filter_.process_noise_spectral_amplitude == 3.0
    # Continuous white acceleration has Q_vv proportional to dt, not dt^2.
    assert filter_.Q[3, 3] == pytest.approx(3.0**2 * 0.5)
