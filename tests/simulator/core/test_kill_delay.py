"""Stochastic kill delay: a lethal hit defers the target's death (death spiral)."""

from types import SimpleNamespace

from bvr_marl_core.simulator.core.events import UnitDestroyedEvent
from bvr_marl_core.simulator.core.hit_event_helpers import (
    KILL_DELAY_MAX_S,
    mark_mortally_hit,
)


class _Sim:
    def __init__(self, delay):
        self.elapsed_time_s = 100.0
        self.rnd_gen = SimpleNamespace(expovariate=lambda _lam: delay)


def test_mark_mortally_hit_schedules_delayed_death():
    tgt = SimpleNamespace(radar_emitting=True)
    missile = SimpleNamespace(id=5)
    sim = _Sim(delay=3.0)
    mark_mortally_hit(tgt, missile, sim)
    assert tgt.is_mortally_hit is True
    assert tgt._death_killer is missile
    assert tgt._death_time_s == 103.0
    assert tgt.radar_emitting is False  # out of the fight (stops radiating)


def test_delay_is_capped():
    tgt = SimpleNamespace()
    sim = _Sim(delay=1e9)  # huge draw
    mark_mortally_hit(tgt, SimpleNamespace(id=1), sim)
    assert tgt._death_time_s == sim.elapsed_time_s + KILL_DELAY_MAX_S


def test_mark_is_idempotent():
    tgt = SimpleNamespace()
    m1, m2 = SimpleNamespace(id=1), SimpleNamespace(id=2)
    sim = _Sim(delay=2.0)
    mark_mortally_hit(tgt, m1, sim)
    first_time = tgt._death_time_s
    mark_mortally_hit(tgt, m2, sim)  # second lethal hit ignored
    assert tgt._death_killer is m1
    assert tgt._death_time_s == first_time


# --- end-to-end through the sim -------------------------------------------

from bvr_marl_core.registry import get_aircraft_class, get_missile_class  # noqa: E402
from bvr_marl_core.simulator.core.helpers import Position  # noqa: E402
from bvr_marl_core.simulator.simulator import Simulator  # noqa: E402

_ML = SimpleNamespace(left_lon=-5, right_lon=5, bottom_lat=-5, top_lat=5, min_alt=0, max_alt=20000)


def _ef(lat, lon, yaw, grp):
    return get_aircraft_class("Eurofighter")(
        Position(lat=lat, lon=lon, alt=9000.0), yaw, 300.0, grp, _ML, 0.0, 20000.0
    )


def test_lethal_hit_defers_death_and_credits_at_death():
    sim = Simulator(tick_secs=0.5, random_seed=3)
    shooter = _ef(0.0, 0.0, 0.0, "BLUE")
    tgt = _ef(0.25, 0.0, 180.0, "RED")
    sim.add_unit(shooter)
    sim.add_unit(tgt)
    for _ in range(6):
        sim.do_tick()
    msl, _, _ = shooter.weapons.fire_missile_direct(sim, tgt, get_missile_class("amraam"))
    assert msl is not None

    hit_alt = None
    death_event = None
    for _ in range(120):
        sim.do_tick()
        if tgt.is_mortally_hit and hit_alt is None:
            hit_alt = tgt.position.alt
            assert tgt.id in sim.active_units, "target must linger after the lethal hit"
            assert tgt.radar_emitting is False
        for e in sim.events:
            if isinstance(e, UnitDestroyedEvent) and e.unit_destroyed is tgt:
                death_event = e
        if tgt.id not in sim.active_units:
            break

    assert hit_alt is not None, "target should have been mortally hit"
    assert death_event is not None, "a kill must be credited when the death fires"
    # Killer attribution survives (missile object persists after removal).
    assert death_event.unit_killer.source.id == shooter.id
    assert tgt.removal_reason == "missile_kill"
    assert tgt.position.alt < hit_alt  # spiraled down before dying


def test_mortally_hit_aircraft_cannot_fire():
    s = _ef(0, 0, 0, "BLUE")
    t = _ef(0.2, 0, 180, "RED")
    sim = Simulator(tick_secs=0.5)
    sim.add_unit(s)
    sim.add_unit(t)
    s.is_mortally_hit = True
    msl, veto, _ = s.weapons.fire_missile_direct(sim, t, get_missile_class("amraam"))
    assert msl is None and veto == "shooter_mortally_hit"
