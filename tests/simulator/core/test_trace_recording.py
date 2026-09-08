"""The per-tick unit trace (visualization-only) can be disabled for headless runs."""

from types import SimpleNamespace

from bvr_marl_core.registry import get_aircraft_class
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator

_ML = SimpleNamespace(left_lon=-5, right_lon=5, bottom_lat=-5, top_lat=5, min_alt=0, max_alt=20000)


def _ef():
    return get_aircraft_class("Eurofighter")(
        Position(lat=0.0, lon=0.0, alt=9000.0), 90.0, 250.0, "BLUE", _ML, 0.0, 20000.0
    )


def test_traces_recorded_by_default():
    sim = Simulator(tick_secs=1.0)
    uid = sim.add_unit(_ef())
    sim.record_unit_trace(uid)
    for _ in range(5):
        sim.do_tick()
    assert uid in sim.trace_record_units
    assert len(sim.trace_record_units[uid]) >= 5


def test_traces_suppressed_when_disabled():
    sim = Simulator(tick_secs=1.0)
    sim.record_traces = False
    uid = sim.add_unit(_ef())
    sim.record_unit_trace(uid)  # no-op
    for _ in range(5):
        sim.do_tick()
    # Nothing stored, so no unbounded growth.
    assert sim.trace_record_units == {}
