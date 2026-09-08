"""Scripted EMCON sensing baselines."""

import numpy as np
import pytest

from bvr_marl_core.aircraft.systems.emcon_controller import EmconController


def test_learned_is_no_override():
    c = EmconController("learned")
    assert not c.is_override
    assert c.emitting(step=0) is None


def test_always_on_off():
    on = EmconController("always_on")
    off = EmconController("always_off")
    assert on.is_override and off.is_override
    assert all(on.emitting(step=s) is True for s in range(5))
    assert all(off.emitting(step=s) is False for s in range(5))


def test_periodic_duty_cycle():
    c = EmconController("periodic", period_steps=10, duty=0.3)
    emit = [c.emitting(step=s) for s in range(20)]
    # 3 of every 10 steps emit -> 30% duty.
    assert sum(emit[:10]) == 3
    assert emit[:3] == [True, True, True] and emit[3:10] == [False] * 7


def test_random_matches_duty_in_expectation():
    c = EmconController("random", duty=0.4)
    rng = np.random.default_rng(0)
    emits = [c.emitting(step=s, rng=rng) for s in range(4000)]
    assert 0.36 < np.mean(emits) < 0.44


def test_heuristic_emits_when_no_tracks():
    c = EmconController("heuristic")

    class _Sensor:
        sensor_tracks = ()

    class _Unit:
        sensor = _Sensor()

    assert c.emitting(step=0, unit=_Unit()) is True


def test_heuristic_silent_when_fresh_and_precise():
    c = EmconController("heuristic", heuristic_age_s=6.0, heuristic_cov_trace=5e6)

    class _Track:
        age_s = 1.0
        covariance = np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])

    class _Sensor:
        sensor_tracks = (_Track(),)

    class _Unit:
        sensor = _Sensor()

    assert c.emitting(step=0, unit=_Unit()) is False


def test_heuristic_emits_when_stale():
    c = EmconController("heuristic", heuristic_age_s=6.0)

    class _Track:
        age_s = 12.0
        covariance = np.eye(6)

    class _Unit:
        class sensor:  # noqa: N801
            sensor_tracks = (_Track(),)

    assert c.emitting(step=0, unit=_Unit()) is True


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        EmconController("telepathic_radar")
