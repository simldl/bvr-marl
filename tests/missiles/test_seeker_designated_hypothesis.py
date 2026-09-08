"""A seeker that loses its designated aircraft must coast, not adopt a neighbour.

The seeker stamps its weapon-track identity onto the measurement nearest the current
guidance point. With no consistency gate that is a relabelling: once the designated
aircraft leaves the seeker gate, the nearest remaining contact -- a wingman, a
different formation member -- inherits the weapon's track identity and the shot
silently transfers to an aircraft nobody engaged.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.radar.core.utils import enu_to_geodetic
from bvr_marl_core.radar.lock.missile import MissileLockController
from bvr_marl_core.radar.units.missile import _DESIGNATED_GATE_CEILING_M, MissileRadar
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator
from bvr_marl_core.simulator.utils.geodesics import geodetic_distance_km

pytestmark = pytest.mark.slow

_KM_PER_DEG = 111.0
# Within this of an aircraft, an estimate is that aircraft rather than a coast.
_ADOPTED_KM = 2.0


class _MapLimits:
    bottom_lat = -5.0
    top_lat = 5.0
    left_lon = -5.0
    right_lon = 5.0
    min_alt = 0.0
    max_alt = 20_000.0


def _fighter(lat, lon, yaw, group, alt=8_000.0):
    return Eurofighter(
        Position(lat, lon, alt),
        yaw_deg=yaw,
        speed_mps=280.0,
        group=group,
        map_limits=_MapLimits(),
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )


def _distance_km(position, unit):
    return geodetic_distance_km(
        position.lat,
        position.lon,
        position.alt,
        unit.position.lat,
        unit.position.lon,
        unit.position.alt,
    )


def _track_positions(missile):
    """Every seeker track estimate as a geodetic position, keyed by track id."""
    origin = missile.position
    positions = {}
    for track in missile.radar.cached_tracks or ():
        lat, lon, alt = enu_to_geodetic(
            np.asarray(track.state[:3], dtype=float), origin.lat, origin.lon, origin.alt
        )
        positions[track.track_id] = Position(lat, lon, alt)
    return positions


def _adopted_units(missile, units):
    """Which of ``units`` a seeker track has settled on (not merely pointed near)."""
    return {
        unit.id
        for position in _track_positions(missile).values()
        for unit in units
        if _distance_km(position, unit) <= _ADOPTED_KM
    }


def _guidance_adopted(missile, units):
    aim = missile.target_provider.get_guidance_target()
    if aim is None:
        return set()
    return {unit.id for unit in units if _distance_km(aim, unit) <= _ADOPTED_KM}


def test_seeker_coasts_designated_hypothesis_and_reacquires_it_after_track_retirement():
    sim = Simulator(tick_secs=0.5, random_seed=5)
    sim.record_traces = False

    shooter = _fighter(0.0, 0.0, 90.0, "blue")
    target = _fighter(0.0, 25.0 / _KM_PER_DEG, 270.0, "red")
    # A genuine second contact that stays in the seeker's field of regard throughout.
    distractor = _fighter(0.10, 20.0 / _KM_PER_DEG, 270.0, "red")
    for unit in (shooter, target, distractor):
        sim.add_unit(unit)

    for _ in range(6):
        sim.do_tick()

    missile, veto, _ = shooter.weapons.fire_missile_direct(sim, target, AIM120_AMRAAM)
    assert missile is not None, veto
    units = (target, distractor)

    for _ in range(6):
        sim.do_tick()
    assert missile.id in sim.active_units
    assert _adopted_units(missile, units) == {target.id}, "seeker never settled on its target"
    designated_before = missile.weapon_track.snapshot.track_id

    # The engaged aircraft leaves the seeker gate; the distractor stays in view.
    departed = Position(3.0, 3.0, 8_000.0)
    target.position.lat, target.position.lon = departed.lat, departed.lon

    coasting_track_retired = False
    for _ in range(10):
        sim.do_tick()
        if missile.id not in sim.active_units:
            break
        assert distractor.id not in _adopted_units(missile, units), (
            "the seeker adopted the distractor as its designated track"
        )
        assert distractor.id not in _guidance_adopted(missile, units), (
            "guidance transferred to an aircraft that was never engaged"
        )
        if designated_before not in _track_positions(missile):
            coasting_track_retired = True

    assert missile.id in sim.active_units, "missile left the sim before re-acquisition"
    assert coasting_track_retired, "precondition: the designated track was never retired"

    # The engaged aircraft returns to the gate, ahead of the missile.
    missile_position = missile.position
    target.position.lat = missile_position.lat
    target.position.lon = missile_position.lon + 8.0 / _KM_PER_DEG
    target.position.alt = missile_position.alt

    reacquired = False
    for _ in range(12):
        sim.do_tick()
        if missile.id not in sim.active_units:
            break
        assert distractor.id not in _adopted_units(missile, units), (
            "the seeker adopted the distractor while re-acquiring"
        )
        if target.id in _adopted_units(missile, units):
            reacquired = True
            break

    assert reacquired, "the seeker never re-acquired its designated aircraft"


def _seeker_with_guidance_at(distance_m):
    """A seeker whose designated estimate sits ``distance_m`` from the measurement."""
    from types import SimpleNamespace

    from bvr_marl_core.radar.units.missile import MissileRadar

    radar = MissileRadar.__new__(MissileRadar)
    radar.a_res_deg = 10.0
    radar.r_res_m = 5_000.0
    radar._time_since_designated_target_seen = 0.0
    radar.owner = SimpleNamespace(
        weapon_track=SimpleNamespace(snapshot=SimpleNamespace(track_id="weapon-track"))
    )
    aim = Position(0.0, 0.0, 8_000.0)
    measured = Position(0.0, (distance_m / 1000.0) / _KM_PER_DEG, 8_000.0)
    radar.target_provider = SimpleNamespace(get_guidance_target=lambda: aim)
    measurement = {
        "d": 20_000.0,
        "measurement_position": measured,
        "measurement_ref": aim,
    }
    return radar, measurement


def test_seeker_does_not_hand_its_weapon_identity_to_a_foreign_measurement():
    """A stamped measurement is gated ~6x more loosely by the tracker (100.0 vs 16.266).

    Stamping whichever measurement is merely nearest therefore hands a neighbouring
    aircraft a privileged path onto the weapon's own track, which is precisely what
    ``retarget_policy == "track_only"`` is supposed to forbid.
    """
    radar, measurement = _seeker_with_guidance_at(30_000.0)

    radar._stamp_designated_measurement([measurement])

    assert "preferred_track_id" not in measurement


def test_seeker_claims_a_measurement_consistent_with_its_designated_estimate():
    radar, measurement = _seeker_with_guidance_at(500.0)

    radar._stamp_designated_measurement([measurement])

    assert measurement["preferred_track_id"] == "weapon-track"


def test_designated_gate_opens_while_coasting_but_stays_bounded():
    radar, measurement = _seeker_with_guidance_at(1.0)

    fresh = radar._designated_association_gate_m(measurement)
    radar._time_since_designated_target_seen = 4.0
    coasted = radar._designated_association_gate_m(measurement)
    radar._time_since_designated_target_seen = 600.0
    long_coast = radar._designated_association_gate_m(measurement)

    assert fresh < _DESIGNATED_GATE_CEILING_M, "the ceiling must not swallow the gate"
    assert coasted > fresh
    assert long_coast == pytest.approx(_DESIGNATED_GATE_CEILING_M)


def test_track_only_lock_controller_never_confirms_a_foreign_track():
    radar = MissileRadar.__new__(MissileRadar)
    radar.owner = SimpleNamespace(
        weapon_track=SimpleNamespace(snapshot=SimpleNamespace(track_id="designated")),
        retarget_policy="track_only",
    )
    radar.lock_ctrl = MissileLockController()
    radar.lock_ctrl.set_mode("track", "designated")
    radar.locked_target = "designated"
    radar.mode = "track"
    radar._time_since_designated_target_seen = 10.0
    radar._designated_target_loss_threshold_s = 5.0
    distractor = SimpleNamespace(
        track_id="distractor",
        engageable=True,
        classification="aircraft",
        confidence=0.9,
        source_ids=(1,),
        state=(1_000.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    for _ in range(6):
        radar._update_lock_ctrl([distractor])
        assert radar.locked_target != "distractor"

    assert "distractor" not in radar.lock_ctrl.locked_target_ids()
