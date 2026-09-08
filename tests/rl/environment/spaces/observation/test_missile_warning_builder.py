"""Tests for MissileWarningBuilder threat-urgency encoding.

The per-warning "threat level" slot (formerly a hardcoded 1.0) encodes
incoming-missile urgency from range + closing rate, so the policy can perceive
the timing of a last-ditch break, not just its direction. Each warning slot is
now [urgency, sin(bearing), cos(bearing)] (WARN_FEATURE_DIM=3).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from bvr_marl_core.rl.environment.spaces.observation.missile_warning_builder import (
    MissileWarningBuilder,
)
from bvr_marl_core.simulator.core.units import Velocity
from tests.helpers.track_snapshot import track_snapshot


def _builder():
    cfg = SimpleNamespace(em_slots=2, warn_sectors=4)
    return MissileWarningBuilder(simulator=None, config=cfg)


def _unit(lat=0.0, lon=0.0, alt=10000.0, vel=Velocity(0.0, 0.0, 0.0)):
    return SimpleNamespace(
        position=SimpleNamespace(lat=lat, lon=lon, alt=alt), velocity=vel, yaw_deg=0.0
    )


def test_urgency_high_for_close_fast_closing_missile():
    b = _builder()
    unit = _unit()
    # ~1.1 km east, flying due west at 300 m/s -> tti ~3.7 s -> high urgency
    missile = _unit(lon=0.01, vel=Velocity(vx=-300.0, vy=0.0, vz=0.0))
    assert b._threat_urgency(unit, missile) > 0.9


def test_urgency_zero_for_non_closing_missile():
    b = _builder()
    unit = _unit()
    # Missile east of unit and flying further east -> not closing -> urgency 0
    missile = _unit(lon=0.01, vel=Velocity(vx=300.0, vy=0.0, vz=0.0))
    assert b._threat_urgency(unit, missile) == pytest.approx(0.0)


def test_urgency_lower_for_distant_missile():
    b = _builder()
    unit = _unit()
    near = _unit(lon=0.02, vel=Velocity(vx=-300.0, vy=0.0, vz=0.0))  # ~2.2 km
    far = _unit(lon=0.20, vel=Velocity(vx=-300.0, vy=0.0, vz=0.0))  # ~22 km
    assert b._threat_urgency(unit, near) > b._threat_urgency(unit, far)


def test_urgency_in_unit_range():
    b = _builder()
    unit = _unit()
    for lon in (0.005, 0.05, 0.5):
        for vx in (-600.0, -50.0, 50.0, 600.0):
            u = b._threat_urgency(unit, _unit(lon=lon, vel=Velocity(vx=vx, vy=0.0, vz=0.0)))
            assert 0.0 <= u <= 1.0


class _ForbiddenActiveUnits:
    def __getattribute__(self, name):
        raise AssertionError(f"simulator truth was accessed: {name}")


def test_sensor_limited_warning_uses_track_estimate_not_active_units():
    cfg = SimpleNamespace(em_slots=2, information_mode="sensor_limited")
    simulator = SimpleNamespace(active_units=_ForbiddenActiveUnits())
    builder = MissileWarningBuilder(simulator, cfg)
    track = track_snapshot(
        71,
        state=(1000.0, 0.0, 0.0, -300.0, 0.0, 0.0),
        classification="missile",
    )
    unit = SimpleNamespace(
        id=1,
        yaw_deg=0.0,
        velocity=Velocity(0.0, 0.0, 0.0),
        sensor=SimpleNamespace(
            sensor_tracks=[track],
            missile_warner=SimpleNamespace(get_current_warning_ids=lambda: [999]),
        ),
    )

    tokens = builder.build(unit)

    assert tokens[0, -1] == pytest.approx(1.0)
    assert tokens[0, 0] > 0.9
    assert tokens[0, 1] == pytest.approx(1.0)


def test_sensor_limited_unlocated_warning_does_not_invent_direction():
    cfg = SimpleNamespace(em_slots=1, information_mode="sensor_limited")
    builder = MissileWarningBuilder(SimpleNamespace(), cfg)
    unit = SimpleNamespace(
        id=1,
        sensor=SimpleNamespace(
            sensor_tracks=[],
            missile_warner=SimpleNamespace(get_current_warning_ids=lambda: [9]),
        ),
    )

    token = builder.build(unit)[0]

    assert token.tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])
