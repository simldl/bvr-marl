"""Missiles: full-datalink vs shooter-only mid-course, and Augmented PN."""

from types import SimpleNamespace

import numpy as np

from bvr_marl_core.missiles.fox3.r37m import R37M
from bvr_marl_core.missiles.fox3.r77_1 import R77_1
from bvr_marl_core.registry import get_aircraft_class, get_missile_class
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator

_ML = SimpleNamespace(left_lon=-5, right_lon=5, bottom_lat=-5, top_lat=5, min_alt=0, max_alt=20000)


def _ef(lat, lon, yaw, grp):
    return get_aircraft_class("Eurofighter")(
        Position(lat=lat, lon=lon, alt=9000.0), yaw, 300.0, grp, _ML, 0.0, 20000.0
    )


def _fire(missile_key):
    sim = Simulator(tick_secs=0.5)
    shooter = _ef(0.0, 0.0, 0.0, "BLUE")
    wing = _ef(0.0, 0.3, 0.0, "BLUE")
    tgt = _ef(0.45, 0.0, 180.0, "RED")
    for u in (shooter, wing, tgt):
        sim.add_unit(u)
    # Three confirmations require three completed physical scan revisits.  The
    # 1 s dwell must not accelerate merely because this scenario uses 0.5 s ticks.
    for _ in range(12):
        sim.do_tick()
    msl, veto, _ = shooter.weapons.fire_missile(sim, tgt, get_missile_class(missile_key))
    assert msl is not None, veto
    for _ in range(4):
        sim.do_tick()
    return sim, shooter, msl


# --- full datalink vs shooter-only mid-course ------------------------------


def test_classic_loses_midcourse_when_shooter_dies():
    sim, shooter, msl = _fire("amraam")
    assert len(msl.radar.get_default_group_radars(sim)) == 1  # shooter supports
    sim.active_units.pop(shooter.id, None)  # shooter destroyed
    assert len(msl.radar.get_default_group_radars(sim)) == 0  # no mid-course support


def test_full_datalink_survives_shooter_death_via_network():
    sim, shooter, msl = _fire("aim260")
    assert len(msl.radar.get_default_group_radars(sim)) == 2  # whole network
    sim.active_units.pop(shooter.id, None)
    # Wingman still supports the full-datalink missile.
    assert len(msl.radar.get_default_group_radars(sim)) == 1


def test_classic_loses_midcourse_when_shooter_turns_cold():
    sim, shooter, msl = _fire("amraam")
    assert len(msl.radar.get_default_group_radars(sim)) == 1
    shooter.control.set_yaw_deg(180.0)  # turn away -> lose lock / break cone
    for _ in range(20):
        sim.do_tick()
    assert len(msl.radar.get_default_group_radars(sim)) == 0


# --- Augmented PN ----------------------------------------------------------


def test_apn_flag_wiring():
    s = _ef(0, 0, 0, "BLUE")
    t = _ef(1, 0, 180, "RED")
    assert get_missile_class("amraam")(0.0, t, s, _ML).use_apn is False
    apn_classes = [get_missile_class(k) for k in ("meteor", "aim260", "k77m")] + [R77_1, R37M]
    for missile_cls in apn_classes:
        m = missile_cls(0.0, t, s, _ML)
        assert m.use_apn is True
        assert m.guidance.pn.use_apn is True


def test_apn_gate_ignores_gentle_maneuver_but_fires_on_hard():
    s = _ef(0, 0, 0, "BLUE")
    t = _ef(1, 0, 180, "RED")
    pn = get_missile_class("meteor")(0.0, t, s, _ML).guidance.pn

    # Constant velocity between calls -> no acceleration -> gated to zero.
    v = np.array([0.0, 250.0, 0.0])
    pn._estimate_target_accel(v, 0.5, True)
    a = pn._estimate_target_accel(v, 0.5, True)
    assert np.linalg.norm(a) == 0.0

    # A large sustained lateral velocity change -> above-threshold accel returned.
    pn2 = get_missile_class("meteor")(0.0, t, s, _ML).guidance.pn
    prev = np.array([0.0, 250.0, 0.0])
    out = np.zeros(3)
    for _ in range(10):
        prev = prev + np.array([120.0, 0.0, 0.0])  # ~24g per 0.5s step of eastward accel
        out = pn2._estimate_target_accel(prev, 0.5, True)
    assert np.linalg.norm(out) >= pn2._APN_MIN_ACCEL
