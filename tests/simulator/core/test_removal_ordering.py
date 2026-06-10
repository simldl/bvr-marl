"""Regression tests for missile removal ordering in Simulator.do_tick.

A missile that reaches its target on the very tick it also trips a removal
condition (out of energy / lifetime / boundary) must still register the hit.
Missiles now flag themselves for *deferred* removal (should_be_removed) instead
of deleting themselves mid-update, so the substepper runs the proximity check
while the missile is still in the sim, and the simulator sweeps flagged units
afterwards (emitting a removal event with a reason).
"""

from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.simulator.core.events import UnitRemovedEvent
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator
from bvr_marl_core.simulator.utils.map_limits import MapLimits


def _map_limits():
    return MapLimits(
        bottom_lat=-2.0,
        top_lat=2.0,
        left_lon=-2.0,
        right_lon=2.0,
        min_alt=0.0,
        max_alt=20000.0,
    )


def _expiring_passing_missile(target, source, ml):
    """An AMRAAM that flies *through* the target this tick (so the CPA is a real
    proximity hit) while its lifetime also expires on the same tick — exercising
    the deferred-removal ordering (hit must register before the removal sweep)."""
    m = AIM120_AMRAAM(firing_time_s=0.0, target=target, source=source, map_limits=ml, group="A")
    m.position = Position(0.0, -0.006, 10000.0)  # ~670 m west of target at (0,0)
    m.yaw_deg = 90.0
    m.desired_yaw_deg = 90.0
    m.speed = 800.0  # crosses the target within one 1 s tick (CPA mid-tick)
    m.life_time_s = 0.5  # elapsed (1 s) >= lifetime -> flagged this tick
    m.arming_time_s = 0.0  # fuze live (isolate from the arming gate)
    # Deterministic kill on contact: flat Pk (no range falloff) so this test
    # isolates removal ordering from the kill-probability model.
    m.hit_probability = 1.0
    m.lethal_radius_m = 0.0
    return m


def test_same_tick_removal_still_registers_hit():
    ml = _map_limits()
    sim = Simulator(tick_secs=1.0, random_seed=1, weapon_config={"missile_hit_radius_m": 500.0})

    shooter = Eurofighter(
        Position(0.0, -0.5, 10000.0), 90.0, 300.0, "A", ml, min_alt_m=0.0, max_alt_m=20000.0
    )
    target = Eurofighter(
        Position(0.0, 0.0, 10000.0), 90.0, 150.0, "B", ml, min_alt_m=0.0, max_alt_m=20000.0
    )
    sim.add_unit(shooter)
    tid = sim.add_unit(target)

    m = _expiring_passing_missile(target, shooter, ml)
    mid = sim.add_unit(m)

    sim.do_tick()

    # The proximity hit must be registered even though the missile also tripped a
    # removal condition (lifetime) on the same tick.
    assert tid not in sim.active_units, "target should be destroyed by the proximity hit"
    assert mid not in sim.active_units, "missile is consumed on hit"


def test_energy_depleted_missile_removed_via_deferred_event():
    ml = _map_limits()
    sim = Simulator(tick_secs=1.0, random_seed=1, weapon_config={"missile_hit_radius_m": 500.0})

    shooter = Eurofighter(
        Position(0.0, -0.5, 10000.0), 90.0, 300.0, "A", ml, min_alt_m=0.0, max_alt_m=20000.0
    )
    target = Eurofighter(
        Position(0.0, 0.0, 10000.0), 90.0, 150.0, "B", ml, min_alt_m=0.0, max_alt_m=20000.0
    )
    sim.add_unit(shooter)
    tid = sim.add_unit(target)

    # Spent missile far from any target -> no hit, removed via the deferred sweep.
    m = AIM120_AMRAAM(firing_time_s=0.0, target=target, source=shooter, map_limits=ml, group="A")
    m.position = Position(0.0, 1.5, 10000.0)  # ~167 km away, well outside the gate
    m.speed = 40.0
    m.engine.fuel_s = 0.0
    m.physics.constant_engine_F = 0.0
    mid = sim.add_unit(m)

    events = sim.do_tick()

    assert mid not in sim.active_units, "spent missile should be removed"
    assert tid in sim.active_units, "distant target is untouched"
    removed = [
        e
        for e in events
        if isinstance(e, UnitRemovedEvent) and getattr(e.removed_unit, "id", None) == mid
    ]
    assert removed, "a UnitRemovedEvent should be emitted for the spent missile"
    assert removed[0].reason == "energy_depleted"
