from types import SimpleNamespace

from bvr_marl_core.simulator.core.events import (
    MissileEngagementEvent,
    UnitDestroyedEvent,
    UnitRegisteredEvent,
)
from bvr_marl_core.simulator.simulator import Simulator


def test_event_ids_are_monotonic_and_reset_per_replay():
    sim = Simulator(random_seed=3)
    unit = SimpleNamespace(id=1)
    first = UnitRegisteredEvent(sim, unit)
    second = MissileEngagementEvent(sim, unit, SimpleNamespace(id=2))
    assert (first.event_id, second.event_id) == (1, 2)
    assert first.time_s == 0.0

    sim.reset_sim({})
    assert UnitRegisteredEvent(sim, unit).event_id == 1


def test_damage_event_has_explicit_cause():
    sim = Simulator()
    missile = SimpleNamespace(is_missile=True)
    event = UnitDestroyedEvent(sim, missile, SimpleNamespace())
    assert event.cause == "missile"
    assert event.event_id == 1
