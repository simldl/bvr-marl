import pytest

from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.domain.information_mode import audit_sensor_limited_simulator
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator

pytestmark = pytest.mark.slow


class _MapLimits:
    bottom_lat = -5.0
    top_lat = 5.0
    left_lon = -5.0
    right_lon = 5.0
    min_alt = 0.0
    max_alt = 20_000.0


def _fighter(lon: float, yaw_deg: float, group: str) -> Eurofighter:
    return Eurofighter(
        Position(0.0, lon, 8_000.0),
        yaw_deg=yaw_deg,
        speed_mps=300.0,
        group=group,
        map_limits=_MapLimits(),
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )


def test_complete_sensor_and_weapon_episode_retains_no_operational_truth_handle():
    sim = Simulator(tick_secs=1.0, random_seed=23)
    shooter = _fighter(0.0, 90.0, "blue")
    target = _fighter(30.0 / 111.0, 270.0, "red")
    sim.add_unit(shooter)
    sim.add_unit(target)

    for _ in range(6):
        sim.do_tick()
        audit_sensor_limited_simulator(sim)

    missile, veto, _ = shooter.weapons.fire_missile_direct(sim, target, AIM120_AMRAAM)
    assert missile is not None, veto
    assert missile.target is None

    for _ in range(20):
        sim.do_tick()
        audit_sensor_limited_simulator(sim)
