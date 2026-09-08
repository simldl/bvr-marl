import math

import numpy as np
import pytest

from bvr_marl_core.radar.obs.observation import RadarObsGenerator
from bvr_marl_core.radar.radar import Radar
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.core.units import Velocity


class DummyLUT:
    freq_hz = 1e9

    def get_probability(self, dist, rcs):
        return 1.0


def make_target(lat=0.0, lon=0.0, alt=1000.0, rcs=1.0):
    T = type("Tgt", (), {})()
    T.position = type("Pos", (), {})()
    T.position.lat, T.position.lon, T.position.alt = lat, lon, alt
    T.rcs = rcs
    T.orientation = 0.0
    T.velocity = np.zeros(3)
    return T


@pytest.fixture
def obsgen():
    return RadarObsGenerator(
        horizontal_fov_deg=90,
        vertical_fov_deg=60,
        max_range_m=5000,
        lut=DummyLUT(),
        snr_threshold_db=0.0,
        false_alarm_rate=0.0,
        np_rng=np.random.default_rng(0),
    )


def test_generate_one_detection(obsgen):
    own_pos = make_target()
    target = make_target(lat=0.01)
    dets = obsgen.generate(own_pos.position, [target], yaw_deg=0.0, pitch_deg=0.0)
    assert len(dets) == 1
    assert "az" in dets[0] and "el" in dets[0]
    assert "T" not in dets[0]
    assert obsgen.last_detection_targets == (target,)


def _notch_obsgen():
    return RadarObsGenerator(
        horizontal_fov_deg=120,
        vertical_fov_deg=90,
        max_range_m=50000,
        lut=DummyLUT(),
        snr_threshold_db=0.0,
        false_alarm_rate=0.0,
        np_rng=np.random.default_rng(0),
        notch_velocity_mps=50.0,
    )


def test_notch_suppresses_beaming_target():
    """A target flying perpendicular to the LOS (range rate ~ 0) is hidden in the
    Doppler notch and produces no detection."""
    obs = _notch_obsgen()
    own = make_target()  # stationary radar at (0, 0)
    own_vel = Velocity(0.0, 0.0, 0.0)
    target = make_target(lat=0.01)  # due north -> LOS r_hat = (E,N,U) = (0,1,0)
    target.velocity = Velocity(vx=300.0, vy=0.0, vz=0.0)  # pure-east = beaming

    dets = obs.generate(own.position, [target], 0.0, 0.0, own_velocity=own_vel)
    assert dets == []


def test_notch_passes_closing_target():
    """A target with strong LOS closure is well outside the notch and detected."""
    obs = _notch_obsgen()
    own = make_target()
    own_vel = Velocity(0.0, 0.0, 0.0)
    target = make_target(lat=0.01)
    target.velocity = Velocity(vx=0.0, vy=-300.0, vz=0.0)  # heading south, closing

    dets = obs.generate(own.position, [target], 0.0, 0.0, own_velocity=own_vel)
    assert len(dets) == 1


def test_notch_disabled_detects_beaming_target():
    """With the notch disabled (default), a beaming target is still detected."""
    own = make_target()
    target = make_target(lat=0.01)
    target.velocity = Velocity(vx=300.0, vy=0.0, vz=0.0)
    obs = RadarObsGenerator(
        horizontal_fov_deg=120,
        vertical_fov_deg=90,
        max_range_m=50000,
        lut=DummyLUT(),
        snr_threshold_db=0.0,
        np_rng=np.random.default_rng(0),
    )
    dets = obs.generate(own.position, [target], 0.0, 0.0, own_velocity=Velocity(0.0, 0.0, 0.0))
    assert len(dets) == 1


def test_measurement_noise_disabled_is_exact():
    """With no measurement noise configured, the report equals the truth."""
    obs = _notch_obsgen()  # no meas-noise args -> 0
    assert obs._apply_measurement_noise(30.0, 5.0, 10_000.0) == (30.0, 5.0, 10_000.0)


def test_measurement_noise_std_matches_config():
    """Reported az/el/d jitter with the configured Gaussian std."""
    obs = RadarObsGenerator(
        horizontal_fov_deg=120,
        vertical_fov_deg=90,
        max_range_m=200_000,
        lut=DummyLUT(),
        snr_threshold_db=0.0,
        np_rng=np.random.default_rng(0),
        meas_angular_noise_deg=0.5,
        meas_range_noise_m=300.0,
    )
    azs, ds = [], []
    for _ in range(4000):
        az, _el, d = obs._apply_measurement_noise(0.0, 0.0, 10_000.0)
        azs.append(az)
        ds.append(d)
    assert abs(float(np.std(azs)) - 0.5) < 0.05
    assert abs(float(np.std(ds)) - 300.0) < 30.0


def test_noise_cross_range_error_grows_with_range():
    """Angular noise -> cross-range error that scales with range (AWACS effect)."""
    obs = RadarObsGenerator(
        horizontal_fov_deg=120,
        vertical_fov_deg=90,
        max_range_m=200_000,
        lut=DummyLUT(),
        snr_threshold_db=0.0,
        np_rng=np.random.default_rng(0),
        meas_angular_noise_deg=0.5,
    )
    sig = math.radians(0.5)

    def cross_range_std(rng_m):
        errs = [
            math.radians(obs._apply_measurement_noise(0.0, 0.0, rng_m)[0]) * rng_m
            for _ in range(4000)
        ]
        return float(np.std(errs))

    near = cross_range_std(10_000.0)
    far = cross_range_std(100_000.0)
    assert far > near * 5  # ~10x range -> ~10x cross-range error
    assert abs(far - sig * 100_000.0) < 0.15 * (sig * 100_000.0)


def test_notch_factor_ramps_with_range_rate():
    """Factor is 0 at zero closure, ramps quadratically, and saturates at 1."""
    obs = _notch_obsgen()  # notch half-width 50 m/s, LOS due north
    dE, dN, dU, dist = 0.0, 1000.0, 0.0, 1000.0

    def factor_for(vy):
        tgt = make_target()
        tgt.velocity = Velocity(0.0, vy, 0.0)
        return obs._notch_detection_factor(dE, dN, dU, dist, tgt, Velocity(0.0, 0.0, 0.0))

    assert factor_for(0.0) == pytest.approx(0.0)
    assert factor_for(25.0) == pytest.approx((25.0 / 50.0) ** 2)
    assert factor_for(50.0) == pytest.approx(1.0)
    assert factor_for(500.0) == pytest.approx(1.0)


@pytest.fixture
def radar_position():
    return Position(lat=0.0, lon=0.0, alt=1_000.0)


@pytest.fixture
def deterministic_radar():
    return Radar(
        horizontal_fov_deg=60.0,
        vertical_fov_deg=30.0,
        max_range_m=10_000.0,
        radar_frequency_hz=10e9,
        tx_power_w=1e5,
        antenna_gain_db=30.0,
        snr_threshold_db=0.0,
        false_alarm_rate=0.0,
        range_resolution_m=50.0,
        angular_resolution_deg=2.0,
    )


def test_detection_boundaries(monkeypatch, deterministic_radar, radar_position):
    from types import SimpleNamespace

    radar = deterministic_radar

    # Replace the rng with a stub that always returns 0.0 so all in-FOV targets are detected
    class _AlwaysDetect:
        def random(self):
            return 0.0

        def uniform(self, low, high):
            return low

    monkeypatch.setattr(radar.obsgen, "np_rng", _AlwaysDetect())

    # 1. Innerhalb Reichweite und FOV
    target = SimpleNamespace(
        position=Position(
            lat=radar_position.lat + 0.005, lon=radar_position.lon, alt=radar_position.alt
        ),
        orientation=0.0,
        rcs=2.0,
        velocity=np.zeros(3),
    )
    dets = radar.obsgen.generate(radar_position, [target], yaw_deg=0.0, pitch_deg=0.0)
    assert len(dets) == 1

    # 2. Outside radar range
    over_range_lat = radar_position.lat + (radar.max_range_m + 1) / 111_000.0
    target.position = Position(lat=over_range_lat, lon=radar_position.lon, alt=radar_position.alt)
    dets = radar.obsgen.generate(radar_position, [target], yaw_deg=0.0, pitch_deg=0.0)
    assert len(dets) == 0

    # 3. Outside azimuth FOV (right side)
    angle = radar.h_fov_deg / 2
    dist = 5000.0
    az_rad = np.radians(angle + 0.1)
    x_east = np.sin(az_rad) * dist
    y_north = np.cos(az_rad) * dist
    delta_lat = y_north / 111_000
    delta_lon = x_east / (111_000 * np.cos(np.radians(radar_position.lat)))
    target.position = Position(
        lat=radar_position.lat + delta_lat,
        lon=radar_position.lon + delta_lon,
        alt=radar_position.alt,
    )
    dets = radar.obsgen.generate(radar_position, [target], yaw_deg=0.0, pitch_deg=0.0)
    assert len(dets) == 0

    # 4. Genau am linken Rand (leicht innerhalb)
    angle = -radar.h_fov_deg / 2 + 0.1
    az_rad = np.radians(angle)
    x_east = np.sin(az_rad) * dist
    y_north = np.cos(az_rad) * dist
    delta_lat = y_north / 111_000
    delta_lon = x_east / (111_000 * np.cos(np.radians(radar_position.lat)))
    target.position = Position(
        lat=radar_position.lat + delta_lat,
        lon=radar_position.lon + delta_lon,
        alt=radar_position.alt,
    )
    dets = radar.obsgen.generate(radar_position, [target], yaw_deg=0.0, pitch_deg=0.0)
    assert len(dets) == 1
