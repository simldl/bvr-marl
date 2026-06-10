import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import pytest

from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.radar.core.utils import geodetic_to_enu
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.core.hit_event_helpers import kill_probability
from bvr_marl_core.simulator.simulator import Simulator
from bvr_marl_core.simulator.utils.angles import signed_yaw_deg_diff
from tests.missiles.test_missile_telemetry_csv import (
    SCENARIOS,
    _apply_maneuver,
    _build_scenario,
    _dist_positions_m,
    _pos,
    _target_in_fov,
    _track_missed_updates,
    _unit_by_id,
    _warmup,
)

pytestmark = pytest.mark.slow


FIELDNAMES = [
    "scenario",
    "seed",
    "tick",
    "time_s",
    "team_size",
    "missile_id",
    "missile_alive",
    "target_id",
    "target_alive",
    "radar_lock_id",
    "provider_target_id",
    "missile_target_id",
    "guidance_tid",
    "coast_guidance_tid",
    "has_fresh_track",
    "has_coastable_track",
    "track_age_s",
    "track_missed_updates",
    "flight_phase",
    "last_guidance_phase",
    "last_pn_subphase",
    "terminal_range_m",
    "terminal_track_grace_s",
    "terminal_track_grace_range_m",
    "guidance_samples_this_tick",
    "true_range_m",
    "guidance_range_m",
    "range_error_m",
    "min_true_range_m",
    "min_guidance_range_m",
    "true_closing_mps",
    "guidance_closing_mps",
    "true_tgo_s",
    "guidance_tgo_s",
    "min_true_tgo_s",
    "true_los_rate_deg_s",
    "guidance_los_rate_deg_s",
    "max_true_los_rate_deg_s",
    "terminal_by_true_range",
    "terminal_by_guidance_range",
    "terminal_by_true_tgo",
    "terminal_by_guidance_tgo",
    "terminal_by_true_any_tick",
    "terminal_by_guidance_any_tick",
    "terminal_mode_any_tick",
    "terminal_mode_last",
    "missed_terminal_by_true_range",
    "missed_terminal_by_guidance_range",
    "missed_terminal_by_true_range_last",
    "missed_terminal_by_guidance_range_last",
    "missed_terminal_by_true_tgo_last",
    "missed_terminal_by_guidance_tgo_last",
    "target_in_seeker_fov",
    "seeker_az_rel_deg",
    "seeker_el_rel_deg",
    "seeker_range_m",
    "missile_yaw_deg",
    "desired_yaw_deg",
    "last_command_yaw_deg",
    "yaw_command_delta_deg",
    "yaw_command_rate_demand_deg_s",
    "achieved_yaw_rate_deg_s",
    "max_yaw_rate_deg_s",
    "yaw_saturation_ratio",
    "yaw_command_saturated",
    "missile_pitch_deg",
    "desired_pitch_deg",
    "last_command_pitch_deg",
    "pitch_command_delta_deg",
    "pitch_command_rate_demand_deg_s",
    "achieved_pitch_rate_deg_s",
    "max_pitch_rate_deg_s",
    "pitch_saturation_ratio",
    "pitch_command_saturated",
    "missile_speed_mps",
    "target_yaw_deg",
    "target_yaw_rate_deg_s",
    "target_speed_mps",
    "target_lateral_accel_est_mps2",
    "target_id_mismatch",
    "detonation_this_tick",
    "detonation_target_id",
    "detonation_miss_m",
    "detonation_pk",
    "cpa_time_s",
    "cpa_miss_m",
    "cpa_t_frac",
]


def _csv_path() -> Path:
    configured = os.environ.get("BVR_MISSILE_ENDGAME_CSV")
    if configured:
        return Path(configured)
    return Path("output") / "missile_telemetry" / "missile_endgame_diagnostics.csv"


def _velocity_enu(unit) -> np.ndarray:
    yaw = math.radians(float(getattr(unit, "yaw_deg", 0.0)))
    pitch = math.radians(float(getattr(unit, "pitch_deg", 0.0)))
    speed = float(getattr(unit, "speed", 0.0))
    cp = math.cos(pitch)
    return np.array(
        [
            cp * math.sin(yaw) * speed,
            cp * math.cos(yaw) * speed,
            math.sin(pitch) * speed,
        ],
        dtype=float,
    )


def _range_kinematics(missile, target_pos: Position | None, target_velocity=None) -> dict:
    if target_pos is None:
        return {}

    try:
        r_vec = np.asarray(
            geodetic_to_enu(
                target_pos.lat,
                target_pos.lon,
                target_pos.alt,
                missile.position.lat,
                missile.position.lon,
                missile.position.alt,
            ),
            dtype=float,
        )
    except Exception:
        return {}

    rng = float(np.linalg.norm(r_vec))
    if not np.isfinite(rng) or rng < 1e-9:
        return {"range_m": rng}

    vm = _velocity_enu(missile)
    vt = np.zeros(3, dtype=float)
    if target_velocity is not None:
        try:
            vt = np.asarray(target_velocity, dtype=float).reshape(3)
        except Exception:
            vt = np.zeros(3, dtype=float)

    vrel = vt - vm
    u_los = r_vec / rng
    closing = -float(np.dot(vrel, u_los))
    los_rate_rad_s = float(np.linalg.norm(np.cross(r_vec, vrel)) / max(rng * rng, 1e-9))
    tgo = rng / closing if closing > 1e-6 else None
    return {
        "range_m": rng,
        "closing_mps": closing,
        "los_rate_deg_s": math.degrees(los_rate_rad_s),
        "tgo_s": tgo,
    }


def _tracker_guidance_state(missile, guidance_pos: Position | None):
    if guidance_pos is None:
        return None, None
    try:
        return missile.guidance.pn._get_target_state_enu(missile.position, guidance_pos)
    except Exception:
        return None, None


def _guidance_kinematics(missile, target, dt: float) -> dict:
    target_pos = _pos(target)
    true = _range_kinematics(missile, target_pos, _velocity_enu(target) if target else None)

    try:
        guidance_pos = missile.target_provider.get_guidance_target()
    except Exception:
        guidance_pos = None

    tracker_p, tracker_v = _tracker_guidance_state(missile, guidance_pos)
    guidance = {}
    if tracker_p is not None:
        rng = float(np.linalg.norm(tracker_p))
        vm = _velocity_enu(missile)
        vt = tracker_v if tracker_v is not None else np.zeros(3, dtype=float)
        if rng > 1e-9:
            u_los = tracker_p / rng
            vrel = vt - vm
            closing = -float(np.dot(vrel, u_los))
            los_rate = float(np.linalg.norm(np.cross(tracker_p, vrel)) / max(rng * rng, 1e-9))
            guidance = {
                "range_m": rng,
                "closing_mps": closing,
                "los_rate_deg_s": math.degrees(los_rate),
                "tgo_s": rng / closing if closing > 1e-6 else None,
            }
    else:
        if guidance_pos is not None:
            guidance = _range_kinematics(missile, guidance_pos, tracker_v)

    terminal_range = float(getattr(missile.guidance.pn, "terminal_range_m", 2000.0))
    true_range = true.get("range_m")
    guidance_range = guidance.get("range_m")
    true_tgo = true.get("tgo_s")
    guidance_tgo = guidance.get("tgo_s")

    return {
        "dt": dt,
        "true_range_m": true_range,
        "guidance_range_m": guidance_range,
        "true_closing_mps": true.get("closing_mps"),
        "guidance_closing_mps": guidance.get("closing_mps"),
        "true_tgo_s": true_tgo,
        "guidance_tgo_s": guidance_tgo,
        "true_los_rate_deg_s": true.get("los_rate_deg_s"),
        "guidance_los_rate_deg_s": guidance.get("los_rate_deg_s"),
        "terminal_range_m": terminal_range,
        "terminal_by_true_range": true_range is not None and true_range < terminal_range,
        "terminal_by_guidance_range": guidance_range is not None
        and guidance_range < terminal_range,
        "terminal_by_true_tgo": true_tgo is not None and true_tgo < 1.5,
        "terminal_by_guidance_tgo": guidance_tgo is not None and guidance_tgo < 1.5,
        "pre_yaw_deg": float(getattr(missile, "yaw_deg", 0.0)),
        "pre_pitch_deg": float(getattr(missile, "pitch_deg", 0.0)),
        "pre_speed_mps": float(getattr(missile, "speed", 0.0)),
        "max_yaw_rate_deg_s": _max_yaw_rate_deg_s(missile),
        "max_pitch_rate_deg_s": float(getattr(missile.physics, "max_pitch_rate_deg_s", 20.0)),
    }


def _max_yaw_rate_deg_s(missile) -> float | None:
    try:
        yaw = float(getattr(missile, "yaw_deg", 0.0))
        _, omega = missile.physics.update_yaw_deg(
            missile.position,
            yaw,
            yaw + 180.0,
            float(getattr(missile, "speed", 0.0)),
            1.0,
        )
        return abs(math.degrees(float(omega)))
    except Exception:
        return None


def _safe_min(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return min(values) if values else None


def _safe_max(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return max(values) if values else None


def _last(samples: list[dict], key: str):
    if not samples:
        return None
    return samples[-1].get(key)


def _any(samples: list[dict], key: str) -> bool:
    return any(bool(sample.get(key)) for sample in samples)


def _instrument_missile(missile, target) -> None:
    missile._endgame_tick_samples = []
    missile._endgame_last_phase = None

    guidance = missile.guidance
    original_select = guidance._select_guidance_phase
    original_compute = guidance.compute_guidance

    def record_select(m, provider, tracker_manager, dt):
        phase = original_select(m, provider, tracker_manager, dt)
        m._endgame_last_phase = phase
        return phase

    def record_compute(m, provider, tracker_manager, dt):
        sample = _guidance_kinematics(m, target, dt)
        yaw, pitch = original_compute(m, provider, tracker_manager, dt)
        phase = getattr(m, "_endgame_last_phase", None)
        sample["guidance_phase"] = phase
        sample["command_yaw_deg"] = yaw
        sample["command_pitch_deg"] = pitch
        sample["yaw_command_delta_deg"] = signed_yaw_deg_diff(sample["pre_yaw_deg"], yaw)
        sample["pitch_command_delta_deg"] = pitch - sample["pre_pitch_deg"]
        sample["yaw_command_rate_demand_deg_s"] = abs(sample["yaw_command_delta_deg"]) / dt
        sample["pitch_command_rate_demand_deg_s"] = abs(sample["pitch_command_delta_deg"]) / dt
        max_yaw = sample.get("max_yaw_rate_deg_s")
        max_pitch = sample.get("max_pitch_rate_deg_s")
        sample["yaw_saturation_ratio"] = (
            sample["yaw_command_rate_demand_deg_s"] / max_yaw
            if max_yaw and max_yaw > 1e-9
            else None
        )
        sample["pitch_saturation_ratio"] = (
            sample["pitch_command_rate_demand_deg_s"] / max_pitch
            if max_pitch and max_pitch > 1e-9
            else None
        )
        sample["yaw_command_saturated"] = (
            sample["yaw_saturation_ratio"] is not None and sample["yaw_saturation_ratio"] > 1.0
        )
        sample["pitch_command_saturated"] = (
            sample["pitch_saturation_ratio"] is not None and sample["pitch_saturation_ratio"] > 1.0
        )
        sample["pn_subphase"] = (
            "terminal"
            if phase == "pn"
            and sample.get("guidance_range_m") is not None
            and sample["guidance_range_m"] < sample["terminal_range_m"]
            else phase
        )
        m._endgame_tick_samples.append(sample)
        return yaw, pitch

    guidance._select_guidance_phase = record_select
    guidance.compute_guidance = record_compute


def _fire_instrumented_pair(sim: Simulator, shooter, target, ml):
    missile, veto, _ = shooter.weapons.fire_missile_direct(sim, target, AIM120_AMRAAM)
    assert missile is not None, veto
    missile._endgame_launch_target_id = getattr(target, "id", None)
    _instrument_missile(missile, target)
    return missile


def _state(unit) -> dict:
    return {
        "pos": _pos(unit),
        "yaw": float(getattr(unit, "yaw_deg", 0.0)),
        "pitch": float(getattr(unit, "pitch_deg", 0.0)),
        "speed": float(getattr(unit, "speed", 0.0)),
    }


def _make_row(
    scenario,
    seed: int,
    tick: int,
    sim: Simulator,
    missile,
    targets: list,
    pre_states: dict,
    hit_records: dict,
) -> dict:
    target_id = getattr(missile, "_endgame_launch_target_id", None)
    target = _unit_by_id(sim, targets, target_id)
    missile_pos = _pos(missile)
    target_pos = _pos(target)
    samples = getattr(missile, "_endgame_tick_samples", [])
    hit = hit_records.get(missile.id, {})
    cpa = hit.get("cpa") or {}

    radar_lock_id = missile.radar.get_locked_target()
    provider_target_id = getattr(missile.target_provider, "current_target_id", None)
    missile_target_id = getattr(getattr(missile, "target", None), "id", None)
    target_id_mismatch = any(
        tid is not None and tid != target_id
        for tid in (radar_lock_id, provider_target_id, missile_target_id)
    )

    az, el, seeker_range, in_fov = _target_in_fov(missile, target_pos)
    true_range = _dist_positions_m(missile_pos, target_pos)
    guidance_pos = missile.target_provider.get_guidance_target()
    guidance_range = _dist_positions_m(missile_pos, guidance_pos)

    pre_m = pre_states.get(("missile", missile.id), {})
    pre_t = pre_states.get(("target", target_id), {})
    tick_dt = float(scenario.tick_secs)
    achieved_yaw_rate = None
    achieved_pitch_rate = None
    if pre_m:
        achieved_yaw_rate = (
            abs(signed_yaw_deg_diff(pre_m["yaw"], float(getattr(missile, "yaw_deg", 0.0))))
            / tick_dt
        )
        achieved_pitch_rate = (
            abs(float(getattr(missile, "pitch_deg", 0.0)) - pre_m["pitch"]) / tick_dt
        )

    target_yaw_rate = None
    target_lat_accel = None
    if pre_t and target is not None:
        target_yaw_rate = signed_yaw_deg_diff(pre_t["yaw"], float(getattr(target, "yaw_deg", 0.0)))
        target_yaw_rate /= tick_dt
        target_lat_accel = abs(float(getattr(target, "speed", 0.0)) * math.radians(target_yaw_rate))

    coast_tid = missile.target_provider.get_coast_guidance_tid()
    has_coastable = missile.target_provider.has_coastable_track()

    terminal_mode_any = any(sample.get("pn_subphase") == "terminal" for sample in samples)
    terminal_mode_last = _last(samples, "pn_subphase") == "terminal"
    terminal_by_true_any = _any(samples, "terminal_by_true_range") or _any(
        samples, "terminal_by_true_tgo"
    )
    terminal_by_guidance_any = _any(samples, "terminal_by_guidance_range") or _any(
        samples, "terminal_by_guidance_tgo"
    )

    return {
        "scenario": scenario.name,
        "seed": seed,
        "tick": tick,
        "time_s": sim.elapsed_time_s,
        "team_size": scenario.team_size,
        "missile_id": missile.id,
        "missile_alive": missile.id in sim.active_units,
        "target_id": target_id,
        "target_alive": target_id in sim.active_units,
        "radar_lock_id": radar_lock_id,
        "provider_target_id": provider_target_id,
        "missile_target_id": missile_target_id,
        "guidance_tid": missile.target_provider.get_guidance_tid(),
        "coast_guidance_tid": coast_tid,
        "has_fresh_track": missile.target_provider.has_fresh_track(),
        "has_coastable_track": has_coastable,
        "track_age_s": getattr(missile.target_provider, "last_confirmed_track_age_s", None),
        "track_missed_updates": _track_missed_updates(missile),
        "flight_phase": getattr(missile.phase_manager, "current_phase", None),
        "last_guidance_phase": _last(samples, "guidance_phase"),
        "last_pn_subphase": _last(samples, "pn_subphase"),
        "terminal_range_m": _last(samples, "terminal_range_m"),
        "terminal_track_grace_s": getattr(missile.guidance, "terminal_track_grace_s", None),
        "terminal_track_grace_range_m": getattr(
            missile.guidance, "terminal_track_grace_range_m", None
        ),
        "guidance_samples_this_tick": len(samples),
        "true_range_m": true_range,
        "guidance_range_m": guidance_range,
        "range_error_m": _dist_positions_m(guidance_pos, target_pos),
        "min_true_range_m": _safe_min([sample.get("true_range_m") for sample in samples]),
        "min_guidance_range_m": _safe_min([sample.get("guidance_range_m") for sample in samples]),
        "true_closing_mps": _last(samples, "true_closing_mps"),
        "guidance_closing_mps": _last(samples, "guidance_closing_mps"),
        "true_tgo_s": _last(samples, "true_tgo_s"),
        "guidance_tgo_s": _last(samples, "guidance_tgo_s"),
        "min_true_tgo_s": _safe_min([sample.get("true_tgo_s") for sample in samples]),
        "true_los_rate_deg_s": _last(samples, "true_los_rate_deg_s"),
        "guidance_los_rate_deg_s": _last(samples, "guidance_los_rate_deg_s"),
        "max_true_los_rate_deg_s": _safe_max(
            [sample.get("true_los_rate_deg_s") for sample in samples]
        ),
        "terminal_by_true_range": _last(samples, "terminal_by_true_range"),
        "terminal_by_guidance_range": _last(samples, "terminal_by_guidance_range"),
        "terminal_by_true_tgo": _last(samples, "terminal_by_true_tgo"),
        "terminal_by_guidance_tgo": _last(samples, "terminal_by_guidance_tgo"),
        "terminal_by_true_any_tick": terminal_by_true_any,
        "terminal_by_guidance_any_tick": terminal_by_guidance_any,
        "terminal_mode_any_tick": terminal_mode_any,
        "terminal_mode_last": terminal_mode_last,
        "missed_terminal_by_true_range": terminal_by_true_any and not terminal_mode_any,
        "missed_terminal_by_guidance_range": terminal_by_guidance_any and not terminal_mode_any,
        "missed_terminal_by_true_range_last": bool(_last(samples, "terminal_by_true_range"))
        and not terminal_mode_last,
        "missed_terminal_by_guidance_range_last": bool(_last(samples, "terminal_by_guidance_range"))
        and not terminal_mode_last,
        "missed_terminal_by_true_tgo_last": bool(_last(samples, "terminal_by_true_tgo"))
        and not terminal_mode_last,
        "missed_terminal_by_guidance_tgo_last": bool(_last(samples, "terminal_by_guidance_tgo"))
        and not terminal_mode_last,
        "target_in_seeker_fov": in_fov,
        "seeker_az_rel_deg": az,
        "seeker_el_rel_deg": el,
        "seeker_range_m": seeker_range,
        "missile_yaw_deg": getattr(missile, "yaw_deg", None),
        "desired_yaw_deg": getattr(missile, "desired_yaw_deg", None),
        "last_command_yaw_deg": _last(samples, "command_yaw_deg"),
        "yaw_command_delta_deg": _last(samples, "yaw_command_delta_deg"),
        "yaw_command_rate_demand_deg_s": _last(samples, "yaw_command_rate_demand_deg_s"),
        "achieved_yaw_rate_deg_s": achieved_yaw_rate,
        "max_yaw_rate_deg_s": _last(samples, "max_yaw_rate_deg_s"),
        "yaw_saturation_ratio": _last(samples, "yaw_saturation_ratio"),
        "yaw_command_saturated": _last(samples, "yaw_command_saturated"),
        "missile_pitch_deg": getattr(missile, "pitch_deg", None),
        "desired_pitch_deg": getattr(missile, "desired_pitch_deg", None),
        "last_command_pitch_deg": _last(samples, "command_pitch_deg"),
        "pitch_command_delta_deg": _last(samples, "pitch_command_delta_deg"),
        "pitch_command_rate_demand_deg_s": _last(samples, "pitch_command_rate_demand_deg_s"),
        "achieved_pitch_rate_deg_s": achieved_pitch_rate,
        "max_pitch_rate_deg_s": _last(samples, "max_pitch_rate_deg_s"),
        "pitch_saturation_ratio": _last(samples, "pitch_saturation_ratio"),
        "pitch_command_saturated": _last(samples, "pitch_command_saturated"),
        "missile_speed_mps": getattr(missile, "speed", None),
        "target_yaw_deg": getattr(target, "yaw_deg", None),
        "target_yaw_rate_deg_s": target_yaw_rate,
        "target_speed_mps": getattr(target, "speed", None),
        "target_lateral_accel_est_mps2": target_lat_accel,
        "target_id_mismatch": target_id_mismatch,
        "detonation_this_tick": bool(hit),
        "detonation_target_id": hit.get("target_id"),
        "detonation_miss_m": hit.get("miss_m"),
        "detonation_pk": hit.get("pk"),
        "cpa_time_s": cpa.get("time_s"),
        "cpa_miss_m": cpa.get("miss_m"),
        "cpa_t_frac": cpa.get("t_frac"),
    }


def _run_scenario_to_rows(scenario, seed: int, max_ticks: int) -> list[dict]:
    rng = random.Random(seed)
    sim, ml, shooters, targets = _build_scenario(scenario, seed)
    _warmup(sim)
    missiles = [
        _fire_instrumented_pair(sim, shooter, target, ml)
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
        pre_states = {}
        for missile in missiles:
            if missile.id in sim.active_units:
                missile._endgame_tick_samples = []
                pre_states[("missile", missile.id)] = _state(missile)
        for target in targets:
            if target.id in sim.active_units:
                pre_states[("target", target.id)] = _state(target)

        hit_records.clear()
        sim.do_tick()

        for missile in missiles:
            if missile.id in sim.active_units or ("missile", missile.id) in pre_states:
                rows.append(
                    _make_row(
                        scenario,
                        seed,
                        tick,
                        sim,
                        missile,
                        targets,
                        pre_states,
                        hit_records,
                    )
                )

        if all(missile.id not in sim.active_units for missile in missiles):
            break

    return rows


def test_missile_endgame_diagnostics_csv_for_terminal_handoff_analysis():
    runs = int(os.environ.get("BVR_MISSILE_ENDGAME_RUNS", "8"))
    max_ticks = int(os.environ.get("BVR_MISSILE_ENDGAME_MAX_TICKS", "90"))
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
    assert any(row["guidance_samples_this_tick"] for row in rows)
    assert any(row["detonation_this_tick"] for row in rows)
