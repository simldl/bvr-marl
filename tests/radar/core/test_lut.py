import pytest

from bvr_marl_core.radar.core.lut import DetectionLUT


@pytest.fixture
def lut():
    return DetectionLUT(
        freq_hz=10e9,
        tx_power_w=1e5,
        gain=20.0,
        max_range_m=10_000.0,
        snr_threshold_db=0.0,
        max_rcs=50.0,
        rcs_bins=50,
        dist_bins=50,
    )


def test_probability_inside_bounds(lut):
    assert 0.0 <= lut.get_probability(500, 2.0) <= 1.0
    assert 0.0 <= lut.get_probability(10000, 50.0) <= 1.0


def test_probability_above_max_rcs_clamps_but_keeps_range_dependence(lut):
    near = lut.get_probability(500, 1e9)
    far = lut.get_probability(9000, 1e9)
    assert near == pytest.approx(lut.get_probability(500, 50.0))
    assert far == pytest.approx(lut.get_probability(9000, 50.0))
    assert far < near


def test_probability_decreases_with_distance(lut):
    p1 = lut.get_probability(500, 10.0)
    p2 = lut.get_probability(9000, 10.0)
    assert p2 < p1
