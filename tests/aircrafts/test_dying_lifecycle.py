from types import SimpleNamespace

from bvr_marl_core.registry import get_aircraft_class
from bvr_marl_core.simulator.core.helpers import Position


def test_dying_aircraft_skips_operational_sensor_update(monkeypatch):
    limits = SimpleNamespace(left_lon=-1, right_lon=1, bottom_lat=-1, top_lat=1)
    aircraft = get_aircraft_class("Eurofighter")(
        Position(0.0, 0.0, 5000.0), 0.0, 250.0, "BLUE", limits, 0.0, 20_000.0
    )
    aircraft.id = 1
    aircraft.is_mortally_hit = True
    aircraft._death_time_s = 100.0
    called = False

    def sensor_update(*_args):
        nonlocal called
        called = True

    monkeypatch.setattr(aircraft.sensor, "update_sensor_data", sensor_update)
    aircraft.update(1.0, SimpleNamespace(elapsed_time_s=0.0))
    assert called is False
