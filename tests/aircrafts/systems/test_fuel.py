"""Aircraft fuel (mass coupling, depletion, flame-out) and missile propellant burn."""

from types import SimpleNamespace

from bvr_marl_core.registry import get_aircraft_class, get_missile_class
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator

_ML = SimpleNamespace(left_lon=-5, right_lon=5, bottom_lat=-5, top_lat=5, min_alt=0, max_alt=20000)


def _ef(alt=9000.0):
    return get_aircraft_class("Eurofighter")(
        Position(lat=0.0, lon=0.0, alt=alt), 0.0, 300.0, "BLUE", _ML, 0.0, 20000.0
    )


def test_full_fuel_mass_equals_config_mass():
    a = _ef()
    # A full tank reproduces the configured (fixed-mass) value at spawn.
    assert a.physics.mass_kg == a.fuel.empty_mass_kg + a.fuel.capacity_kg
    assert a.fuel.fuel_fraction == 1.0
    assert a.fuel.enabled is True


def test_fuel_depletes_and_mass_drops():
    a = _ef()
    sim = Simulator(tick_secs=1.0)
    sim.add_unit(a)
    a.control.set_throttle(1.0)
    m0 = a.physics.mass_kg
    f0 = a.fuel.fuel_kg
    for _ in range(120):
        sim.do_tick()
    assert a.fuel.fuel_kg < f0
    assert a.physics.mass_kg < m0
    # The force model (which drives Ps / induced drag) tracks the current mass.
    assert a.physics.force_model.mass_kg == a.physics.mass_kg
    assert a.physics.mass_kg == a.fuel.empty_mass_kg + a.fuel.fuel_kg


def test_fuel_disabled_keeps_mass_fixed():
    # Build an aircraft with fuel disabled (capacity 0) via a raw config.
    from bvr_marl_core.aircraft.aircraft import Aircraft

    cfg = {"mass_kg": 12000.0, "fuel_capacity_kg": 0.0, "max_speed_mps": 600.0}
    a = Aircraft("t", Position(0, 0, 9000), 0.0, 300.0, "BLUE", _ML, cfg)
    assert a.fuel.enabled is False
    assert a.fuel.fuel_fraction == 1.0
    m0 = a.physics.mass_kg
    a.fuel.consume(1e5, 10.0)  # no-op when disabled
    assert a.physics.mass_kg == m0


def test_flame_out_glides_and_crashes():
    a = _ef(alt=9000.0)
    a.fuel.fuel_kg = 5.0  # nearly empty
    sim = Simulator(tick_secs=1.0)
    sim.add_unit(a)
    a.control.set_throttle(1.0)
    for _ in range(600):
        sim.do_tick()
        if a.should_be_removed:
            break
    assert a.fuel.flamed_out is True
    assert a.should_be_removed is True
    assert a.removal_reason == "out_of_fuel"
    assert a.position.alt <= a.min_alt_m + 60.0  # descended to the floor


def test_missile_propellant_mass_burns_off():
    shooter = _ef()
    tgt = _ef()
    tgt.group = "RED"
    msl = get_missile_class("amraam")(0.0, tgt, shooter, _ML)
    assert msl.launch_mass_kg == 160.0
    assert msl.burnout_mass_kg < msl.launch_mass_kg
    assert msl.physics.mass_kg == msl.launch_mass_kg

    # Burn to completion; mass should fall to burnout and stay there.
    for _ in range(int(msl.engine.initial_fuel_s) + 5):
        msl.engine.update(1.0)
    assert abs(msl.physics.mass_kg - msl.burnout_mass_kg) < 1e-6
    assert msl.physics.force_model.mass_kg == msl.physics.mass_kg


def test_fuel_fraction_in_own_observation():
    from bvr_marl_core.rl.environment.spaces.observation.constants import d_OWN
    from bvr_marl_core.rl.environment.spaces.observation.own_state_builder import OwnStateBuilder

    a = _ef()
    a.fuel.fuel_kg = 0.5 * a.fuel.capacity_kg
    sim = Simulator(tick_secs=1.0)
    uid = sim.add_unit(a)
    builder = OwnStateBuilder(sim, SimpleNamespace())
    state = builder.build(sim.active_units[uid])
    assert len(state) == d_OWN == 16
    # [14] fuel fraction, [15] station time (see constants.d_OWN).
    assert abs(float(state[14]) - 0.5) < 1e-3
    # Station time normalized by ~1 h; with fuel remaining it is in (0, 1].
    assert 0.0 < float(state[15]) <= 1.0


def test_station_time_scales_with_fuel_and_flags_disabled():
    from bvr_marl_core.aircraft.aircraft import Aircraft

    a = _ef()
    full = a.fuel.station_time_s
    assert full > 0.0
    a.fuel.fuel_kg = 0.5 * a.fuel.capacity_kg
    half = a.fuel.station_time_s
    # Less fuel -> less endurance (and it is a realistic tens-of-minutes figure).
    assert half < full
    assert 300.0 < full < 3.0 * 3600.0

    # Fuel not modelled -> unlimited endurance sentinel.
    cfg = {"mass_kg": 12000.0, "fuel_capacity_kg": 0.0, "max_speed_mps": 600.0}
    b = Aircraft("t", Position(0, 0, 9000), 0.0, 300.0, "BLUE", _ML, cfg)
    assert b.fuel.station_time_s >= 1e9
