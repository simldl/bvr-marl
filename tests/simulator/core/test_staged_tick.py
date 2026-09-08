from dataclasses import dataclass, field

from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.core.units import Unit
from bvr_marl_core.simulator.simulator import Simulator


@dataclass
class ObservingMover(Unit):
    observations: list[tuple] = field(default_factory=list)

    def update(self, tick_secs, sim):
        self.observations.append(
            tuple(
                sorted(
                    (unit.name, unit.position.lon)
                    for unit in sim.active_units.values()
                    if unit.id != self.id
                )
            )
        )
        self.position.lon += tick_secs
        return []


@dataclass
class SpawnOnce(Unit):
    spawned: bool = False

    def update(self, tick_secs, sim):
        if not self.spawned:
            sim.add_unit(ObservingMover("child", Position(0.0, 0.0, 0.0), 0.0, 0.0))
            self.spawned = True
        return []


@dataclass
class StagedSensorMover(Unit):
    reports_ready: bool = False
    fused_views: list[tuple[str, ...]] = field(default_factory=list)

    def stage_sensor_reports(self, tick_secs, sim):
        self.reports_ready = True

    def update_staged_sensor_products(self, tick_secs, sim):
        self.fused_views.append(
            tuple(
                sorted(
                    unit.name
                    for unit in sim.active_units.values()
                    if getattr(unit, "reports_ready", False)
                )
            )
        )

    def update_after_staged_sensors(self, tick_secs, sim):
        self.position.lon += tick_secs
        return []

    def update(self, tick_secs, sim):
        raise AssertionError("globally staged units must not use monolithic update")


def _run_with_ids(first_id, second_id):
    first = ObservingMover("first", Position(0.0, 0.0, 0.0), 0.0, 0.0)
    second = ObservingMover("second", Position(0.0, 0.0, 0.0), 0.0, 0.0)
    first.id = first_id
    second.id = second_id
    simulator = Simulator(tick_secs=1.0)
    simulator.reset_sim({first_id: first, second_id: second})
    simulator.do_tick()
    return first, second


def test_units_observe_same_start_state_independently_of_id_order():
    first_a, second_a = _run_with_ids(1, 2)
    first_b, second_b = _run_with_ids(2, 1)

    assert first_a.observations == first_b.observations == [(("second", 0.0),)]
    assert second_a.observations == second_b.observations == [(("first", 0.0),)]
    assert first_a.position.lon == first_b.position.lon == 1.0
    assert second_a.position.lon == second_b.position.lon == 1.0


def test_mid_tick_creation_is_first_updated_on_next_tick():
    spawner = SpawnOnce("spawner", Position(0.0, 0.0, 0.0), 0.0, 0.0)
    simulator = Simulator(tick_secs=1.0)
    simulator.add_unit(spawner)

    simulator.do_tick()
    child = next(unit for unit in simulator.active_units.values() if unit.name == "child")
    assert child.observations == []

    simulator.do_tick()
    assert len(child.observations) == 1


def test_all_raw_reports_exist_before_any_fusion_independently_of_id_order():
    def run(first_id, second_id):
        first = StagedSensorMover("first", Position(0.0, 0.0, 0.0), 0.0, 0.0)
        second = StagedSensorMover("second", Position(0.0, 0.0, 0.0), 0.0, 0.0)
        first.id, second.id = first_id, second_id
        simulator = Simulator(tick_secs=1.0)
        simulator.reset_sim({first_id: first, second_id: second})
        simulator.do_tick()
        return first.fused_views, second.fused_views

    expected = [("first", "second")]
    first_a, second_a = run(1, 2)
    first_b, second_b = run(2, 1)

    assert first_a == second_a == first_b == second_b == expected
