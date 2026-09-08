from types import SimpleNamespace

import pytest

from bvr_marl_core.aircraft.core.poles import estimate_launch_poles
from bvr_marl_core.missiles.missile_parameters import MissileParameters, MissileRadarParameters
from bvr_marl_core.simulator.utils.geodesics import geodetic_direct


def _pos_at_range(range_m: float):
    lat, lon, alt = geodetic_direct(48.0, 11.0, 10_000.0, 0.0, range_m)
    return SimpleNamespace(lat=lat, lon=lon, alt=alt)


def _missile_params(*, max_range_m: float = 160_000.0, seeker_range_m: float = 40_000.0):
    return MissileParameters(
        name="TestFox3",
        max_range_m=max_range_m,
        min_range_m=1_500.0,
        max_speed_mps=1_300.0,
        max_g=40.0,
        seeker_sensitivity=1.0,
        fox_type=3,
        hit_probability=0.85,
        life_time_s=120.0,
        motor_burn_s=8.0,
        mass_kg=160.0,
        radar=MissileRadarParameters(
            horizontal_fov_deg=60.0,
            vertical_fov_deg=30.0,
            max_range_m=seeker_range_m,
            radar_frequency_hz=10e9,
            tx_power_w=8e3,
            antenna_gain_db=32.0,
            snr_threshold_db=8.0,
        ),
    )


def test_hot_geometry_a_pole_precedes_f_pole():
    own = SimpleNamespace(
        id="own",
        position=_pos_at_range(0.0),
        yaw_deg=0.0,
        speed=400.0,
    )
    target = SimpleNamespace(
        id="target",
        position=_pos_at_range(80_000.0),
        yaw_deg=180.0,
        speed=300.0,
    )

    estimate = estimate_launch_poles(own, target, _missile_params(seeker_range_m=30_000.0))

    assert estimate.valid
    assert estimate.active_supported
    assert estimate.intercept_possible
    assert estimate.time_to_active_s < estimate.time_to_impact_s
    assert estimate.f_pole_range_m < estimate.a_pole_range_m < estimate.slant_range_m
    assert estimate.active_range_m == pytest.approx(30_000.0)


def test_cold_shooter_can_open_range_before_intercept():
    own = SimpleNamespace(
        id="own",
        position=_pos_at_range(0.0),
        yaw_deg=180.0,
        speed=400.0,
    )
    target = SimpleNamespace(
        id="target",
        position=_pos_at_range(80_000.0),
        yaw_deg=180.0,
        speed=300.0,
    )

    estimate = estimate_launch_poles(own, target, _missile_params(seeker_range_m=30_000.0))

    assert estimate.valid
    assert estimate.shooter_target_closure_mps < 0.0
    assert estimate.f_pole_range_m > estimate.slant_range_m


def test_active_range_is_capped_to_terminal_handoff_fraction():
    own = SimpleNamespace(
        id="own",
        position=_pos_at_range(0.0),
        yaw_deg=0.0,
        speed=400.0,
    )
    target = SimpleNamespace(
        id="target",
        position=_pos_at_range(80_000.0),
        yaw_deg=180.0,
        speed=300.0,
    )

    estimate = estimate_launch_poles(own, target, _missile_params(seeker_range_m=150_000.0))

    assert estimate.valid
    assert estimate.active_range_m == pytest.approx(40_000.0)
