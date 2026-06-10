import random

import pytest

from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.core.hit_event_helpers import default_on_hit, kill_probability
from bvr_marl_core.simulator.simulator import Simulator

pytestmark = pytest.mark.slow


class _MapLimits:
    bottom_lat = -5.0
    top_lat = 5.0
    left_lon = -5.0
    right_lon = 5.0
    min_alt = 0.0
    max_alt = 20_000.0


def _run_live_like_amraam_beam(seed: int = 0, with_distractor: bool = False) -> dict:
    rng = random.Random(seed)
    ml = _MapLimits()
    sim = Simulator(tick_secs=1.0, random_seed=seed)
    shooter = Eurofighter(
        Position(0.0, 0.0, 8_000.0),
        yaw_deg=90.0,
        speed_mps=300.0,
        group="blue",
        map_limits=ml,
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )
    target = Eurofighter(
        Position(0.0, 18.0 / 111.0, 8_000.0),
        yaw_deg=270.0,
        speed_mps=300.0,
        group="red",
        map_limits=ml,
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )

    sim.add_unit(shooter)
    sim.add_unit(target)

    distractor = None
    if with_distractor:
        distractor = Eurofighter(
            Position(0.035, (18.0 / 111.0) * 0.75, 8_000.0),
            yaw_deg=270.0,
            speed_mps=300.0,
            group="red",
            map_limits=ml,
            min_alt_m=0.0,
            max_alt_m=20_000.0,
        )
        sim.add_unit(distractor)

    for _ in range(6):
        sim.do_tick()

    missile, veto, _ = shooter.weapons.fire_missile_direct(sim, target, AIM120_AMRAAM)
    assert missile is not None, veto

    hit_records = []

    def recording_on_hit(m, tgt, t_frac, s, miss_distance_m=None):
        hit_records.append(
            {
                "target_id": getattr(tgt, "id", None),
                "miss_m": miss_distance_m,
                "pk": kill_probability(m, miss_distance_m),
            }
        )
        return default_on_hit(m, tgt, t_frac, s, miss_distance_m=miss_distance_m)

    sim.ccd.on_hit = recording_on_hit

    identity_history = []
    for tick in range(80):
        target.control.set_yaw_deg(0.0)
        if distractor is not None and distractor.id in sim.active_units:
            # Keep a plausible extra contact in the same fight without placing it
            # directly on the designated target's intercept line.
            distractor.control.set_yaw_deg(270.0 + 8.0 * rng.random())

        if missile.id in sim.active_units:
            identity_history.append(
                {
                    "radar_lock": missile.radar.get_locked_target(),
                    "provider_target": missile.target_provider.current_target_id,
                    "missile_target": getattr(missile.target, "id", None),
                }
            )

        sim.do_tick()
        if missile.id not in sim.active_units:
            break

    return {
        "target_id": target.id,
        "distractor_id": getattr(distractor, "id", None),
        "target_alive": target.id in sim.active_units,
        "distractor_alive": (None if distractor is None else distractor.id in sim.active_units),
        "hit_records": hit_records,
        "identity_history": identity_history,
    }


def test_live_like_one_second_beam_intercepts_with_small_miss_distance():
    result = _run_live_like_amraam_beam(seed=0)

    assert result["hit_records"], "missile never reached proximity fuze detonation"
    detonation = result["hit_records"][-1]
    assert detonation["target_id"] == result["target_id"]
    assert detonation["miss_m"] < 40.0
    assert detonation["pk"] > 0.75


def test_live_like_one_second_beam_keeps_designated_target_with_distractor():
    result = _run_live_like_amraam_beam(seed=1, with_distractor=True)

    assert result["hit_records"], "missile never reached proximity fuze detonation"
    assert result["hit_records"][-1]["target_id"] == result["target_id"]
    assert result["distractor_alive"] is True

    radar_locks = {entry["radar_lock"] for entry in result["identity_history"]}
    provider_targets = {entry["provider_target"] for entry in result["identity_history"]}
    missile_targets = {entry["missile_target"] for entry in result["identity_history"]}

    assert radar_locks == {result["target_id"]}
    assert provider_targets == {result["target_id"]}
    assert missile_targets == {result["target_id"]}


def test_failed_seeker_substep_marks_track_stale_for_guidance():
    ml = _MapLimits()
    sim = Simulator(tick_secs=1.0, random_seed=3)
    shooter = Eurofighter(
        Position(0.0, 0.0, 8_000.0),
        yaw_deg=90.0,
        speed_mps=300.0,
        group="blue",
        map_limits=ml,
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )
    target = Eurofighter(
        Position(0.0, 0.12, 8_000.0),
        yaw_deg=270.0,
        speed_mps=300.0,
        group="red",
        map_limits=ml,
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )
    sim.add_unit(shooter)
    sim.add_unit(target)

    missile = AIM120_AMRAAM(
        firing_time_s=sim.elapsed_time_s,
        target=target,
        source=shooter,
        map_limits=ml,
        group="blue",
    )
    missile.designated_target_id = target.id
    missile.target_provider.current_target_id = target.id
    missile.radar.lock_ctrl.set_mode("track", target.id)
    missile.radar.locked_target = target.id
    sim.add_unit(missile)

    dt = 0.25
    assert missile.radar.update_designated_track_substep(
        dt,
        target,
        missile.position,
        target_position=target.position,
    )
    missile.target_provider.update(sim, dt)
    assert missile.target_provider.has_fresh_track()
    assert missile.target_provider.get_guidance_tid() == target.id

    # New tick: staleness is now per-tick (a track measured earlier in the same
    # tick stays fresh for that tick's remaining substeps). Start a fresh tick so
    # the failed substep coasts and actually ages the track.
    missile.radar.begin_designated_tick()
    out_of_fov = Position(missile.position.lat + 0.02, missile.position.lon, missile.position.alt)
    assert not missile.radar.update_designated_track_substep(
        dt,
        target,
        missile.position,
        target_position=out_of_fov,
    )
    missile.target_provider.update(sim, dt)

    assert missile.target_provider.get_guidance_tid() is None
    assert not missile.target_provider.has_fresh_track()
    assert missile.radar.tracker_manager.tracks[target.id].missed_updates == pytest.approx(dt)

    phase = missile.guidance._select_guidance_phase(
        missile,
        missile.target_provider,
        missile.radar.tracker_manager,
        dt,
    )
    assert phase == "direct"
