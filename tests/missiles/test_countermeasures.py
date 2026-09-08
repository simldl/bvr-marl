"""Countermeasure seduction: chaff (beam-gated), decoy (any aspect), radar-only."""

from types import SimpleNamespace

from bvr_marl_core.domain.information import (
    FrameReference,
    TrackLifecycle,
    TrackSnapshot,
    WeaponTrack,
)
from bvr_marl_core.missiles.countermeasures import (
    CHAFF_NOTCH_MPS,
    _beam_factor,
    evaluate_seduction,
)
from bvr_marl_core.simulator.core.helpers import Position


def _cm(cm_type, age=0.0, life=8.0, uid=99):
    return SimpleNamespace(
        cm_type=cm_type,
        age_s=age,
        lifetime_s=life,
        is_countermeasure=True,
        id=uid,
        position=Position(0.1, 0.0, 9000.0),
    )


def _target(vx, vy, cms):
    return SimpleNamespace(
        position=Position(0.1, 0.0, 9000.0),
        velocity=SimpleNamespace(vx=vx, vy=vy, vz=0.0),
        countermeasures=SimpleNamespace(active_countermeasures=cms),
    )


def _missile(fox_type, target):
    return SimpleNamespace(
        fox_type=fox_type,
        position=Position(0.0, 0.0, 9000.0),  # ~44 km south of target
        target=target,
        seduced_by=None,
    )


class _Sim:
    def __init__(self, roll):
        self.active_units = {}
        self.rnd_gen = SimpleNamespace(random=lambda: roll)


def test_beam_factor_high_when_beaming_zero_when_hot():
    # LOS from missile (south) to target (north) ~ +North unit.
    los_north = (0.0, 1.0, 0.0)
    beaming = SimpleNamespace(vx=300.0, vy=0.0, vz=0.0)  # east -> perpendicular
    hot = SimpleNamespace(vx=0.0, vy=-300.0, vz=0.0)  # toward the missile
    assert _beam_factor(SimpleNamespace(velocity=beaming), los_north) > 0.9
    assert _beam_factor(SimpleNamespace(velocity=hot), los_north) == 0.0
    slow = SimpleNamespace(vx=0.0, vy=-0.5 * CHAFF_NOTCH_MPS, vz=0.0)
    assert 0.4 < _beam_factor(SimpleNamespace(velocity=slow), los_north) < 0.6


def test_chaff_seduces_when_beaming():
    tgt = _target(vx=300.0, vy=0.0, cms=[_cm("chaff")])  # beaming
    m = _missile(3, tgt)  # Fox-3 (radar)
    evaluate_seduction(m, _Sim(roll=0.0), dt=1.0)
    assert m.seduced_by is not None


def test_chaff_does_not_seduce_when_hot():
    tgt = _target(vx=0.0, vy=-300.0, cms=[_cm("chaff")])  # hot (nose-on)
    m = _missile(3, tgt)
    evaluate_seduction(m, _Sim(roll=0.0), dt=1.0)  # even a guaranteed roll can't help
    assert m.seduced_by is None  # beam_factor 0 -> zero seduction rate


def test_decoy_seduces_from_any_aspect():
    tgt = _target(vx=0.0, vy=-300.0, cms=[_cm("decoy", life=15.0)])  # hot
    m = _missile(3, tgt)
    evaluate_seduction(m, _Sim(roll=0.0), dt=1.0)
    assert m.seduced_by is not None


def test_ir_missile_ignores_chaff():
    tgt = _target(vx=300.0, vy=0.0, cms=[_cm("chaff")])  # beaming radar chaff
    m = _missile(2, tgt)  # Fox-2 (IR) — chaff is a radar CM
    evaluate_seduction(m, _Sim(roll=0.0), dt=1.0)
    assert m.seduced_by is None


def test_flare_seduces_ir_missile():
    tgt = _target(vx=0.0, vy=-300.0, cms=[_cm("flare", life=5.0)])  # hot; flares are aspect-free
    m = _missile(2, tgt)  # Fox-2 (IR)
    evaluate_seduction(m, _Sim(roll=0.0), dt=1.0)
    assert m.seduced_by is not None


def test_radar_missile_ignores_flares():
    tgt = _target(vx=0.0, vy=-300.0, cms=[_cm("flare", life=5.0)])
    m = _missile(3, tgt)  # Fox-3 (radar) — flares are IR
    evaluate_seduction(m, _Sim(roll=0.0), dt=1.0)
    assert m.seduced_by is None


def test_out_of_range_not_seduced():
    tgt = _target(vx=300.0, vy=0.0, cms=[_cm("decoy")])
    m = _missile(3, tgt)
    m.position = Position(-1.0, 0.0, 9000.0)  # ~155 km away, beyond seduce range
    evaluate_seduction(m, _Sim(roll=0.0), dt=1.0)
    assert m.seduced_by is None


def test_expired_chaff_has_no_effect():
    tgt = _target(vx=300.0, vy=0.0, cms=[_cm("chaff", age=8.0, life=8.0)])  # window 0
    m = _missile(3, tgt)
    evaluate_seduction(m, _Sim(roll=0.0), dt=1.0)
    assert m.seduced_by is None


def test_launch_spawns_countermeasure_object_in_sim():
    # Regression: the launch path must pass the simulator so the object spawns.
    from bvr_marl_core.registry import get_aircraft_class
    from bvr_marl_core.simulator.simulator import Simulator

    ml = SimpleNamespace(
        left_lon=-5, right_lon=5, bottom_lat=-5, top_lat=5, min_alt=0, max_alt=20000
    )
    ac = get_aircraft_class("Eurofighter")(
        Position(0.0, 0.0, 9000.0), 0.0, 300.0, "BLUE", ml, 0.0, 20000.0
    )
    sim = Simulator(tick_secs=1.0)
    sim.add_unit(ac)
    n_before = len(sim.active_units)
    ac.countermeasures.launch_chaff(sim)
    assert len(sim.active_units) == n_before + 1
    assert any(getattr(u, "cm_type", None) == "chaff" for u in sim.active_units.values())


def test_sensor_limited_weapon_track_is_seduced_without_storing_units():
    target = _target(vx=300.0, vy=0.0, cms=[_cm("chaff")])
    snapshot = TrackSnapshot(
        track_id=7,
        state_time_s=0.0,
        state=(0.0, 44_000.0, 0.0, 0.0, 300.0, 0.0),
        covariance=tuple(tuple(float(i == j) for j in range(6)) for i in range(6)),
        confidence=1.0,
        lifecycle=TrackLifecycle.CONFIRMED,
        engageable=True,
        reference_frame=FrameReference(0.0, 0.0, 9000.0),
    )
    missile = _missile(3, None)
    missile.weapon_track = WeaponTrack(snapshot, launch_time_s=0.0)
    missile.seduced_position = None
    sim = _Sim(roll=0.0)
    sim.evaluator_target_for_weapon = lambda _missile: target

    evaluate_seduction(missile, sim, dt=1.0)

    assert missile.target is None
    assert missile.seduced_by is None
    assert missile.seduced_position == target.countermeasures.active_countermeasures[0].position
