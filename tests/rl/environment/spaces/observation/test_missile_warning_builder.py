"""Tests for MissileWarningBuilder threat-urgency encoding.

The per-warning "threat level" slot (formerly a hardcoded 1.0) now encodes
incoming-missile urgency from range + closing rate, so the policy can perceive
the timing of a last-ditch break, not just its direction. The observation
dimension is unchanged (1 + warn_sectors per slot).
"""

from types import SimpleNamespace

import pytest

from bvr_marl_core.rl.environment.spaces.observation.missile_warning_builder import (
    MissileWarningBuilder,
)
from bvr_marl_core.simulator.core.units import Velocity


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
