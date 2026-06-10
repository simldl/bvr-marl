from types import SimpleNamespace

import numpy as np
import pytest

from bvr_marl_core.radar.core.utils import (
    _angles_dist,
    _doppler,
    _effective_rcs,
    enu_to_geodetic,
    geodetic_to_enu,
    to_cart,
)


class DummyPosition:
    def __init__(self, lat, lon, alt):
        self.lat, self.lon, self.alt = lat, lon, alt


def test_geodetic_enu_inverse():
    pos = DummyPosition(50, 8, 120)
    ref = DummyPosition(50, 8.001, 100)
    enu = geodetic_to_enu(pos.lat, pos.lon, pos.alt, ref.lat, ref.lon, ref.alt)
    lat2, lon2, alt2 = enu_to_geodetic(enu, ref.lat, ref.lon, ref.alt)
    assert np.allclose([pos.lat, pos.lon, pos.alt], [lat2, lon2, alt2], atol=1e-5)


def test_angles_dist():
    p1 = DummyPosition(0, 0, 1000)
    p2 = DummyPosition(0.01, 0, 1000)
    az, el, dist = _angles_dist(p1, 0, 0, p2)
    assert abs(az) < 1e-2
    assert abs(el) < 1e-2
    assert 1100 < dist < 1120


def test_effective_rcs_simple():
    t = SimpleNamespace(position=DummyPosition(0, 0, 0), rcs=10.0)
    eff = _effective_rcs(t, DummyPosition(1, 1, 1))
    assert eff > 0


def test_doppler_shift():
    t = SimpleNamespace(velocity=np.array([100.0, 0.0, 0.0]))
    shift = _doppler(t, az_abs_deg=0, el_abs_deg=0, freq_hz=1e9)
    assert np.isfinite(shift)


def test_doppler_is_exact():
    import numpy as np

    from bvr_marl_core.radar.core.utils import _doppler

    t = type("Target", (), {})()
    t.velocity = np.array([10.0, 0.0, 0.0])  # 10 m/s East in ENU
    freq = 10e9
    expected = (2 * freq * 10.0) / 3e8
    # az=90° points East, so LOS is aligned with velocity
    dop = _doppler(t, az_abs_deg=90, el_abs_deg=0, freq_hz=freq)
    assert np.isclose(dop, expected, rtol=1e-3)


def test_to_cart_is_cartesian():
    # Geodetic azimuth: 0°=North, 90°=East
    # Test North pointing: az=0, el=0 should give (E=0, N=1000, U=0)
    e, n, u = to_cart(0, 0, 1000)
    assert np.isclose(e, 0)  # No east component
    assert np.isclose(n, 1000)  # Points north
    assert np.isclose(u, 0)  # Horizontal

    # Test East pointing: az=90, el=0 should give (E=1000, N=0, U=0)
    e, n, u = to_cart(90, 0, 1000)
    assert np.isclose(e, 1000)  # Points east
    assert np.isclose(n, 0)  # No north component
    assert np.isclose(u, 0)  # Horizontal
