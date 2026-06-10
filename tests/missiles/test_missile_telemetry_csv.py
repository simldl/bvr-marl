import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.radar.core.utils import _angles_dist
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.core.hit_event_helpers import kill_probability
from bvr_marl_core.simulator.simulator import Simulator

pytestmark = pytest.mark.slow


class _MapLimits:
    bottom_lat = -5.0
    top_lat = 5.0
    left_lon = -5.0
    right_lon = 5.0
    min_alt = 0.0
    max_alt = 20_000.0


@dataclass(frozen=True)
class _Scenario:
    name: str
    team_size: int
    maneuver: str
    tick_secs: float = 1.0
    range_km: float = 18.0
    target_speed_mps: float = 300.0
    jink_interval_s: float = 3.0


SCENARIOS = (
    _Scenario("1v1_beam", team_size=1, maneuver="beam"),
    _Scenario("1v1_hard_jink", team_size=1, maneuver="jink"),
    _Scenario("2v2_cross_beam", team_size=2, maneuver="cross_beam"),
    _Scenario("2v2_hard_jink", team_size=2, maneuver="jink"),
)


FIELDNAMES = [
    "scenario",
    "seed",
    "tick",
    "time_s",
    "team_size",
    "missile_id",
    "missile_name",
    "missile_alive",
    "missile_group",
    "launch_target_id",
    "designated_target_id",
    "radar_lock_id",
    "provider_target_id",
    "missile_target_id",
    "guidance_tid",
    "has_fresh_track",
    "track_missed_updates",
    "track_age_s",
    "guidance_phase",
    "flight_phase",
    "missile_lat",
    "missile_lon",
    "missile_alt_m",
    "missile_yaw_deg",
    "missile_pitch_deg",
    "missile_speed_mps",
    "desired_yaw_deg",
    "desired_pitch_deg",
    "fuel_s",
    "target_id",
    "target_alive",
    "target_group",
    "target_lat",
    "target_lon",
    "target_alt_m",
    "target_yaw_deg",
    "target_pitch_deg",
    "target_speed_mps",
    "distance_to_launch_target_m",
    "distance_to_locked_target_m",
    "distance_to_provider_target_m",
    "distance_to_missile_target_m",
    "nearest_enemy_id",
    "nearest_enemy_distance_m",
    "seeker_az_rel_deg",
    "seeker_el_rel_deg",
    "seeker_range_m",
    "target_in_seeker_fov",
    "guidance_position_error_m",
    "ccd_closest_this_tick_m",
    "detonation_this_tick",
    "detonation_target_id",
    "detonation_miss_m",
    "detonation_pk",
    "cpa_time_s",
    "cpa_t_frac",
    "cpa_miss_m",
    "cpa_missile_lat",
    "cpa_missile_lon",
    "cpa_missile_alt_m",
    "cpa_target_lat",
    "cpa_target_lon",
    "cpa_target_alt_m",
    "cpa_target_id",
    "target_id_mismatch",
]


def _csv_path() -> Path:
    configured = os.environ.get("BVR_MISSILE_TELEMETRY_CSV")
    if configured:
        return Path(configured)
    return Path("output") / "missile_telemetry" / "missile_target_telemetry.csv"


def _dist_positions_m(a: Position | None, b: Position | None) -> float | None:
    if a is None or b is None:
        return None
    cos_lat = math.cos(math.radians(0.5 * (a.lat + b.lat)))
    de = (b.lon - a.lon) * 111_000.0 * cos_lat
    dn = (b.lat - a.lat) * 111_000.0
    du = b.alt - a.alt
    return math.sqrt(de * de + dn * dn + du * du)


def _ccd_min_dist(m0: Position, t0: Position, m1: Position, t1: Position) -> float:
    def enu(p: Position, ref: Position) -> np.ndarray:
        cos_lat = math.cos(math.radians(ref.lat))
        return np.array(
            [
                (p.lon - ref.lon) * 111_000.0 * cos_lat,
                (p.lat - ref.lat) * 111_000.0,
                p.alt - ref.alt,
            ],
            dtype=float,
        )

    r0 = -enu(t0, m0)
    vr = enu(m1, m0) - enu(t1, m0)
    denom = float(vr @ vr)
    tau = 0.0 if denom < 1e-9 else max(0.0, min(1.0, float(-(r0 @ vr) / denom)))
    return float(np.linalg.norm(r0 + tau * vr))


def _pos(unit) -> Position | None:
    p = getattr(unit, "position", None)
    if p is None:
        return None
    return Position(float(p.lat), float(p.lon), float(p.alt))


def _build_fighter(lat: float, lon: float, yaw: float, speed: float, group: str, ml):
    return Eurofighter(
        Position(lat, lon, 8_000.0),
        yaw_deg=yaw,
        speed_mps=speed,
        group=group,
        map_limits=ml,
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )


def _build_scenario(scenario: _Scenario, seed: int):
    ml = _MapLimits()
    sim = Simulator(tick_secs=scenario.tick_secs, random_seed=seed)
    range_deg = scenario.range_km / 111.0
    lat_offsets = [0.0] if scenario.team_size == 1 else [-0.025, 0.025]

    shooters = []
    targets = []
    for idx, lat in enumerate(lat_offsets):
        shooter = _build_fighter(lat, 0.0, 90.0, 300.0, "blue", ml)
        target = _build_fighter(
            lat,
            range_deg,
            270.0,
            scenario.target_speed_mps,
            "red",
            ml,
        )
        sim.add_unit(shooter)
        sim.add_unit(target)
        shooters.append(shooter)
        targets.append(target)

    return sim, ml, shooters, targets


def _warmup(sim: Simulator, ticks: int = 6) -> None:
    for _ in range(ticks):
        sim.do_tick()


def _fire_direct_pair(sim: Simulator, shooter, target, ml):
    missile, veto, _ = shooter.weapons.fire_missile_direct(sim, target, AIM120_AMRAAM)
    assert missile is not None, veto
    missile._telemetry_launch_target_id = getattr(target, "id", None)
    missile._telemetry_launch_target = target
    missile._telemetry_guidance_phase = None

    guidance = missile.guidance
    original_select = guidance._select_guidance_phase

    def record_select(m, provider, tracker_manager, dt):
        phase = original_select(m, provider, tracker_manager, dt)
        m._telemetry_guidance_phase = phase
        return phase

    guidance._select_guidance_phase = record_select
    return missile


def _apply_maneuver(scenario: _Scenario, targets: list, tick: int, rng: random.Random) -> None:
    t = tick * scenario.tick_secs
    if scenario.maneuver == "beam":
        for target in targets:
            if not getattr(target, "is_destroyed", False):
                target.control.set_yaw_deg(0.0)
    elif scenario.maneuver == "cross_beam":
        for idx, target in enumerate(targets):
            if not getattr(target, "is_destroyed", False):
                target.control.set_yaw_deg(0.0 if idx == 0 else 180.0)
    elif scenario.maneuver == "jink":
        if t % scenario.jink_interval_s < scenario.tick_secs:
            for target in targets:
                if not getattr(target, "is_destroyed", False):
                    target.control.set_yaw_deg(rng.uniform(0.0, 360.0))


def _unit_by_id(sim: Simulator, units: list, unit_id):
    if unit_id is None:
        return None
    unit = getattr(sim, "active_units", {}).get(unit_id)
    if unit is not None:
        return unit
    for unit in units:
        if getattr(unit, "id", None) == unit_id:
            return unit
    return None


def _track_missed_updates(missile) -> float | None:
    tid = missile.target_provider.get_guidance_tid()
    if tid is None:
        tid = getattr(missile.target_provider, "current_target_id", None)
    tracker = missile.radar.tracker_manager.tracks.get(tid)
    if tracker is None:
        return None
    return float(getattr(tracker, "missed_updates", 0.0))


def _target_in_fov(missile, target_pos: Position | None):
    if target_pos is None:
        return (None, None, None, None)
    try:
        az, el, rng = _angles_dist(
            missile.position,
            float(missile.radar.yaw_deg),
            float(missile.radar.pitch_deg),
            target_pos,
        )
    except Exception:
        return (None, None, None, None)
    in_fov = (
        abs(az) <= float(missile.radar.h_fov_deg) * 0.5
        and abs(el) <= float(missile.radar.v_fov_deg) * 0.5
    )
    return (az, el, rng, in_fov)


def _make_row(
    scenario: _Scenario,
    seed: int,
    tick: int,
    sim: Simulator,
    missile,
    enemies: list,
    pre_positions: dict,
    hit_records: dict,
) -> dict:
    launch_target_id = getattr(missile, "_telemetry_launch_target_id", None)
    launch_target = _unit_by_id(sim, enemies, launch_target_id)
    radar_lock_id = missile.radar.get_locked_target()
    provider_target_id = getattr(missile.target_provider, "current_target_id", None)
    missile_target_id = getattr(getattr(missile, "target", None), "id", None)

    locked_target = _unit_by_id(sim, enemies, radar_lock_id)
    provider_target = _unit_by_id(sim, enemies, provider_target_id)
    missile_target = _unit_by_id(sim, enemies, missile_target_id)

    missile_pos = _pos(missile)
    launch_pos = _pos(launch_target)
    target_pos = launch_pos
    nearest_enemy = None
    nearest_dist = None
    for enemy in enemies:
        d = _dist_positions_m(missile_pos, _pos(enemy))
        if d is not None and (nearest_dist is None or d < nearest_dist):
            nearest_dist = d
            nearest_enemy = enemy

    az, el, seeker_range, in_fov = _target_in_fov(missile, target_pos)
    gp = missile.target_provider.get_guidance_target()
    guidance_err = _dist_positions_m(gp, target_pos)

    ccd_closest = None
    pre_m = pre_positions.get(("missile", missile.id))
    pre_t = pre_positions.get(("target", launch_target_id))
    if (
        pre_m is not None
        and pre_t is not None
        and missile_pos is not None
        and target_pos is not None
    ):
        ccd_closest = _ccd_min_dist(pre_m, pre_t, missile_pos, target_pos)

    hit = hit_records.get(missile.id, {})
    cpa = hit.get("cpa") or {}
    cpa_missile_pose = cpa.get("missile_pose") or (None, None, None)
    cpa_target_pose = cpa.get("target_pose") or (None, None, None)
    target_id_mismatch = any(
        tid is not None and tid != launch_target_id
        for tid in (radar_lock_id, provider_target_id, missile_target_id)
    )

    target_alive = bool(launch_target_id in sim.active_units)
    missile_alive = bool(missile.id in sim.active_units)

    return {
        "scenario": scenario.name,
        "seed": seed,
        "tick": tick,
        "time_s": sim.elapsed_time_s,
        "team_size": scenario.team_size,
        "missile_id": missile.id,
        "missile_name": getattr(missile, "name", type(missile).__name__),
        "missile_alive": missile_alive,
        "missile_group": getattr(missile, "group", None),
        "launch_target_id": launch_target_id,
        "designated_target_id": getattr(missile, "designated_target_id", None),
        "radar_lock_id": radar_lock_id,
        "provider_target_id": provider_target_id,
        "missile_target_id": missile_target_id,
        "guidance_tid": missile.target_provider.get_guidance_tid(),
        "has_fresh_track": missile.target_provider.has_fresh_track(),
        "track_missed_updates": _track_missed_updates(missile),
        "track_age_s": getattr(missile.target_provider, "last_confirmed_track_age_s", None),
        "guidance_phase": getattr(missile, "_telemetry_guidance_phase", None),
        "flight_phase": getattr(missile.phase_manager, "current_phase", None),
        "missile_lat": getattr(missile.position, "lat", None),
        "missile_lon": getattr(missile.position, "lon", None),
        "missile_alt_m": getattr(missile.position, "alt", None),
        "missile_yaw_deg": getattr(missile, "yaw_deg", None),
        "missile_pitch_deg": getattr(missile, "pitch_deg", None),
        "missile_speed_mps": getattr(missile, "speed", None),
        "desired_yaw_deg": getattr(missile, "desired_yaw_deg", None),
        "desired_pitch_deg": getattr(missile, "desired_pitch_deg", None),
        "fuel_s": getattr(getattr(missile, "engine", None), "fuel_s", None),
        "target_id": launch_target_id,
        "target_alive": target_alive,
        "target_group": getattr(launch_target, "group", None),
        "target_lat": getattr(getattr(launch_target, "position", None), "lat", None),
        "target_lon": getattr(getattr(launch_target, "position", None), "lon", None),
        "target_alt_m": getattr(getattr(launch_target, "position", None), "alt", None),
        "target_yaw_deg": getattr(launch_target, "yaw_deg", None),
        "target_pitch_deg": getattr(launch_target, "pitch_deg", None),
        "target_speed_mps": getattr(launch_target, "speed", None),
        "distance_to_launch_target_m": _dist_positions_m(missile_pos, launch_pos),
        "distance_to_locked_target_m": _dist_positions_m(missile_pos, _pos(locked_target)),
        "distance_to_provider_target_m": _dist_positions_m(missile_pos, _pos(provider_target)),
        "distance_to_missile_target_m": _dist_positions_m(missile_pos, _pos(missile_target)),
        "nearest_enemy_id": getattr(nearest_enemy, "id", None),
        "nearest_enemy_distance_m": nearest_dist,
        "seeker_az_rel_deg": az,
        "seeker_el_rel_deg": el,
        "seeker_range_m": seeker_range,
        "target_in_seeker_fov": in_fov,
        "guidance_position_error_m": guidance_err,
        "ccd_closest_this_tick_m": ccd_closest,
        "detonation_this_tick": bool(hit),
        "detonation_target_id": hit.get("target_id"),
        "detonation_miss_m": hit.get("miss_m"),
        "detonation_pk": hit.get("pk"),
        "cpa_time_s": cpa.get("time_s"),
        "cpa_t_frac": cpa.get("t_frac"),
        "cpa_miss_m": cpa.get("miss_m"),
        "cpa_missile_lat": cpa_missile_pose[0],
        "cpa_missile_lon": cpa_missile_pose[1],
        "cpa_missile_alt_m": cpa_missile_pose[2],
        "cpa_target_lat": cpa_target_pose[0],
        "cpa_target_lon": cpa_target_pose[1],
        "cpa_target_alt_m": cpa_target_pose[2],
        "cpa_target_id": cpa.get("target_id"),
        "target_id_mismatch": target_id_mismatch,
    }


def _run_scenario_to_rows(scenario: _Scenario, seed: int, max_ticks: int) -> list[dict]:
    rng = random.Random(seed)
    sim, ml, shooters, targets = _build_scenario(scenario, seed)
    _warmup(sim)
    missiles = [
        _fire_direct_pair(sim, shooter, target, ml)
        for shooter, target in zip(shooters, targets, strict=True)
    ]

    hit_records: dict[int, dict] = {}
    original_on_hit = sim.ccd.on_hit

    def record_on_hit(missile, target, t_frac, s, miss_distance_m=None):
        hit_records[missile.id] = {
            "target_id": getattr(target, "id", None),
            "miss_m": miss_distance_m,
            "pk": kill_probability(missile, miss_distance_m),
            "cpa": getattr(missile, "_last_cpa_event", None),
        }
        return original_on_hit(missile, target, t_frac, s, miss_distance_m=miss_distance_m)

    sim.ccd.on_hit = record_on_hit

    rows = []
    for tick in range(max_ticks):
        _apply_maneuver(scenario, targets, tick, rng)
        pre_positions = {}
        for missile in missiles:
            if missile.id in sim.active_units:
                pre_positions[("missile", missile.id)] = _pos(missile)
        for target in targets:
            if target.id in sim.active_units:
                pre_positions[("target", target.id)] = _pos(target)

        hit_records.clear()
        sim.do_tick()

        for missile in missiles:
            if missile.id in sim.active_units or ("missile", missile.id) in pre_positions:
                rows.append(
                    _make_row(
                        scenario,
                        seed,
                        tick,
                        sim,
                        missile,
                        targets,
                        pre_positions,
                        hit_records,
                    )
                )

        if all(missile.id not in sim.active_units for missile in missiles):
            break

    return rows


def test_missile_target_telemetry_csv_for_1v1_and_2v2_runs():
    runs = int(os.environ.get("BVR_MISSILE_TELEMETRY_RUNS", "8"))
    max_ticks = int(os.environ.get("BVR_MISSILE_TELEMETRY_MAX_TICKS", "90"))
    out_path = _csv_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario in SCENARIOS:
        for seed in range(runs):
            rows.extend(_run_scenario_to_rows(scenario, seed, max_ticks=max_ticks))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    assert out_path.exists()
    assert rows
    assert any(row["team_size"] == 2 for row in rows)
    assert any(row["detonation_this_tick"] for row in rows)
