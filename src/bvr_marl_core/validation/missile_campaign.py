"""Seeded launch-to-terminal validation using production missile dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.missiles.fox2.sidewinder import AIM9_Sidewinder
from bvr_marl_core.missiles.fox3.aim260 import AIM260_JATM
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.simulator.core.events import MissileTerminalEvent
from bvr_marl_core.simulator.core.helpers import Position, units_distance_km
from bvr_marl_core.simulator.simulator import Simulator


class _MapLimits:
    bottom_lat = -5.0
    top_lat = 5.0
    left_lon = -5.0
    right_lon = 5.0
    min_alt = 0.0
    max_alt = 20_000.0


@dataclass(frozen=True, slots=True)
class MissileCase:
    family: str
    name: str
    range_km: float = 30.0
    shooter_alt_m: float = 9_000.0
    shooter_speed_mps: float = 300.0
    target_alt_m: float = 9_000.0
    target_speed_mps: float = 250.0
    aspect: str = "hot"
    maneuver: str = "none"
    missile: str = "amraam"
    countermeasure: str = "none"
    countermeasure_range_km: float = 12.0
    jammer_burn_through_km: float = 0.0
    support: str = "continuous"
    tick_s: float = 0.5
    guidance_reliability: float | None = None
    seeker_reliability: float | None = None
    datalink_reliability: float | None = None
    fuze_reliability: float | None = None
    warhead_effectiveness: float | None = None


_ASPECT_YAW = {"hot": 270.0, "beam": 0.0, "cold": 90.0}
_MISSILES = {
    "amraam": AIM120_AMRAAM,
    "aim260": AIM260_JATM,
    "sidewinder": AIM9_Sidewinder,
}


def _fighter(lat: float, lon: float, altitude_m: float, yaw: float, speed: float, group: str):
    return Eurofighter(
        Position(lat, lon, altitude_m),
        yaw_deg=yaw,
        speed_mps=speed,
        group=group,
        map_limits=_MapLimits(),
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )


def _target_east(shooter, case: MissileCase):
    dz = case.target_alt_m - case.shooter_alt_m
    slant_m = case.range_km * 1_000.0
    horizontal_m = math.sqrt(max(0.0, slant_m**2 - dz**2))
    longitude = shooter.position.lon + horizontal_m / (
        111_000.0 * math.cos(math.radians(shooter.position.lat))
    )
    return _fighter(
        shooter.position.lat,
        longitude,
        case.target_alt_m,
        _ASPECT_YAW[case.aspect],
        case.target_speed_mps,
        "red",
    )


def _deploy_countermeasure(target, simulator: Simulator, kind: str) -> bool:
    system = target.countermeasures
    if kind == "chaff":
        return system.launch_chaff(simulator) is not None
    if kind == "decoy":
        return system.deploy_decoys(simulator) is not None
    if kind == "flare":
        return system.launch_flares(simulator) is not None
    return False


def _terminal_record(simulator: Simulator, missile) -> dict[str, Any] | None:
    records = [
        event.record
        for event in simulator.events
        if isinstance(event, MissileTerminalEvent) and event.missile is missile
    ]
    return records[-1] if records else None


def _run_case(case: MissileCase, seed: int) -> dict[str, Any]:
    simulator = Simulator(tick_secs=case.tick_s, random_seed=seed)
    simulator.record_traces = False
    shooter = _fighter(0.0, 0.0, case.shooter_alt_m, 90.0, case.shooter_speed_mps, "blue")
    target = _target_east(shooter, case)
    target.noise_jammer_burn_through_km = case.jammer_burn_through_km
    simulator.add_unit(shooter)
    simulator.add_unit(target)

    wing = None
    if case.missile == "aim260" or case.support == "network_only":
        wing = _fighter(-0.08, 0.0, case.shooter_alt_m, 90.0, case.shooter_speed_mps, "blue")
        simulator.add_unit(wing)

    missile_class = _MISSILES[case.missile]
    missile, veto, _ = shooter.weapons.fire_missile_direct(simulator, target, missile_class)
    if missile is None:
        raise RuntimeError(f"validation launch vetoed: {veto}")
    for attribute in (
        "guidance_reliability",
        "seeker_reliability",
        "datalink_reliability",
        "fuze_reliability",
        "warhead_effectiveness",
    ):
        sampled_value = getattr(case, attribute)
        if sampled_value is not None:
            setattr(missile, attribute, float(sampled_value))

    launch_position = missile.position.copy()
    first_seeker_report_s: float | None = None
    first_stable_track_s: float | None = None
    minimum_distance_m = float("inf")
    maximum_distance_from_launch_m = 0.0
    countermeasure_deployed = False
    range_denied_reports = 0
    support_interrupted = False
    max_ticks = int(math.ceil(180.0 / case.tick_s))

    for tick in range(max_ticks):
        elapsed_s = tick * case.tick_s
        if case.maneuver == "hard_turn" and tick % max(1, round(2.0 / case.tick_s)) == 0:
            target.control.set_yaw_deg((target.yaw_deg + 55.0) % 360.0)
        elif case.maneuver == "beam" and tick == 0:
            target.control.set_yaw_deg(0.0)

        if (
            not countermeasure_deployed
            and case.countermeasure != "none"
            and units_distance_km(missile, target) <= case.countermeasure_range_km
        ):
            countermeasure_deployed = _deploy_countermeasure(target, simulator, case.countermeasure)

        if case.support in {"turn_cold", "network_only"} and elapsed_s >= 4.0:
            shooter.control.set_yaw_deg(270.0)
            support_interrupted = True

        simulator.do_tick()
        if target.id in simulator.active_units and missile.id in simulator.active_units:
            minimum_distance_m = min(
                minimum_distance_m, units_distance_km(missile, target) * 1_000.0
            )
            maximum_distance_from_launch_m = max(
                maximum_distance_from_launch_m,
                units_distance_km(missile, type("Launch", (), {"position": launch_position})())
                * 1_000.0,
            )
        reports = getattr(missile.radar, "cached_detections", None) or ()
        if reports and first_seeker_report_s is None:
            first_seeker_report_s = (tick + 1) * case.tick_s
        range_denied_reports += sum(bool(report.get("range_denied", False)) for report in reports)
        stable_track = getattr(missile.radar, "has_stable_seeker_track", None)
        if callable(stable_track) and stable_track() and first_stable_track_s is None:
            first_stable_track_s = (tick + 1) * case.tick_s
        elif (
            not callable(stable_track)
            and getattr(missile.radar, "get_locked_target", lambda: None)() is not None
            and first_stable_track_s is None
        ):
            first_stable_track_s = (tick + 1) * case.tick_s
        if missile.id not in simulator.active_units:
            break

    terminal = _terminal_record(simulator, missile)
    lifecycle_complete = missile.id not in simulator.active_units
    return {
        "family": case.family,
        "case": case.name,
        "seed": seed,
        "missile": case.missile,
        "range_km": case.range_km,
        "shooter_alt_m": case.shooter_alt_m,
        "shooter_speed_mps": case.shooter_speed_mps,
        "target_alt_m": case.target_alt_m,
        "target_speed_mps": case.target_speed_mps,
        "aspect": case.aspect,
        "maneuver": case.maneuver,
        "countermeasure": case.countermeasure,
        "jammer_burn_through_km": case.jammer_burn_through_km,
        "support": case.support,
        "tick_s": case.tick_s,
        "guidance_reliability": case.guidance_reliability,
        "seeker_reliability": case.seeker_reliability,
        "datalink_reliability": case.datalink_reliability,
        "fuze_reliability": case.fuze_reliability,
        "warhead_effectiveness": case.warhead_effectiveness,
        "terminal_event": terminal is not None,
        "lifecycle_complete": lifecycle_complete,
        # Detonated / geometric hit / killed are reported separately: a shot can
        # detonate far outside the lethal radius and draw a low Pk, which is a very
        # different failure from a well-placed shot losing its roll.
        "detonated": bool(terminal and terminal.get("detonated")),
        "geometric_hit": bool(terminal and terminal.get("geometric_hit")),
        "killed": bool(terminal and terminal.get("killed")),
        "lethal_radius_m": (terminal or {}).get("lethal_radius_m"),
        "fuze_radius_m": (terminal or {}).get("fuze_radius_m"),
        "pk": float(terminal.get("pk", 0.0)) if terminal else 0.0,
        "miss_distance_m": (
            float(terminal["miss_distance_m"])
            if terminal and terminal.get("miss_distance_m") is not None
            else None
        ),
        "cpa_time_s": (
            float(terminal["cpa_time_s"])
            if terminal and terminal.get("cpa_time_s") is not None
            else None
        ),
        "minimum_sampled_distance_m": minimum_distance_m,
        "maximum_flyout_from_launch_m": maximum_distance_from_launch_m,
        "first_seeker_report_s": first_seeker_report_s,
        "first_stable_track_s": first_stable_track_s,
        "range_denied_report_count": range_denied_reports,
        "countermeasure_deployed": countermeasure_deployed,
        "seduced": getattr(missile, "seduced_position", None) is not None,
        "support_interrupted": support_interrupted,
        "removal_reason": getattr(missile, "removal_reason", None),
    }


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    keys = sorted({(row["family"], row["case"]) for row in rows})
    for family, name in keys:
        subset = [row for row in rows if row["family"] == family and row["case"] == name]
        trials = len(subset)
        terminal = sum(row["terminal_event"] for row in subset)
        detonations = sum(row["detonated"] for row in subset)
        geometric_hits = sum(row["geometric_hit"] for row in subset)
        kills = sum(row["killed"] for row in subset)
        seduced = sum(row["seduced"] for row in subset)
        kill_low, kill_high = _wilson(kills, trials)
        misses = [row["miss_distance_m"] for row in subset if row["miss_distance_m"] is not None]
        seeker = [row["first_seeker_report_s"] for row in subset if row["first_seeker_report_s"]]
        summaries.append(
            {
                "family": family,
                "case": name,
                "shots": trials,
                "terminal_fraction": terminal / trials,
                "detonation_fraction": detonations / trials,
                "geometric_hit_fraction": geometric_hits / trials,
                "kill_fraction": kills / trials,
                # Of the shots that detonated, how many were actually inside the
                # warhead pattern. A low value means the fuze envelope is firing
                # on approaches the warhead cannot convert.
                "geometric_hit_per_detonation": (
                    geometric_hits / detonations if detonations else None
                ),
                "kill_per_geometric_hit": (kills / geometric_hits if geometric_hits else None),
                "kill_ci95_low": kill_low,
                "kill_ci95_high": kill_high,
                "seduction_fraction": seduced / trials,
                "median_miss_distance_m": float(np.median(misses)) if misses else None,
                "median_first_seeker_report_s": float(np.median(seeker)) if seeker else None,
            }
        )
    return summaries


def missile_launch_to_terminal(output: Path, seed: int) -> dict[str, Any]:
    """Run a bounded pairwise campaign over launch, seeker, EW, and timestep axes."""
    geometry_cases = [
        MissileCase(
            "geometry",
            f"{aspect}_{alt // 1000}km",
            aspect=aspect,
            shooter_alt_m=alt,
            target_alt_m=alt,
            range_km=30.0 if alt < 14_000 else 50.0,
        )
        for aspect in ("hot", "beam", "cold")
        for alt in (2_000.0, 9_000.0, 14_000.0)
    ]
    seeker_cases = [
        MissileCase("seeker", "nominal", range_km=35.0),
        MissileCase("seeker", "hard_turn", range_km=35.0, maneuver="hard_turn"),
        MissileCase("seeker", "support_interrupted", range_km=35.0, support="turn_cold"),
        MissileCase(
            "seeker", "network_only", range_km=35.0, missile="aim260", support="network_only"
        ),
    ]
    ew_cases = [
        MissileCase("ew", "no_countermeasure", range_km=25.0, aspect="beam"),
        MissileCase("ew", "beam_chaff", range_km=25.0, aspect="beam", countermeasure="chaff"),
        MissileCase("ew", "hot_chaff", range_km=25.0, aspect="hot", countermeasure="chaff"),
        MissileCase("ew", "hot_decoy", range_km=25.0, aspect="hot", countermeasure="decoy"),
        MissileCase(
            "ew",
            "ir_flare",
            range_km=12.0,
            aspect="hot",
            missile="sidewinder",
            countermeasure="flare",
            countermeasure_range_km=8.0,
        ),
        MissileCase("ew", "classic_jammer", range_km=35.0, jammer_burn_through_km=5.0),
        MissileCase(
            "ew",
            "network_jammer",
            range_km=35.0,
            missile="aim260",
            jammer_burn_through_km=5.0,
            support="network_only",
        ),
    ]
    timestep_cases = [
        MissileCase("timestep", f"dt_{dt:g}", range_km=30.0, maneuver="hard_turn", tick_s=dt)
        for dt in (1.0, 0.5, 0.25)
    ]
    cases = geometry_cases + seeker_cases + ew_cases + timestep_cases
    rows = [
        _run_case(case, seed + case_index * 100 + replicate)
        for case_index, case in enumerate(cases)
        for replicate in range(4)
    ]

    rng = np.random.default_rng(seed + 90_000)
    aspects = ("hot", "beam", "cold")
    countermeasures = ("none", "chaff", "decoy")
    for replicate in range(24):
        altitude = float(rng.uniform(3_000.0, 14_000.0))
        case = MissileCase(
            "uncertainty",
            "randomized",
            range_km=float(rng.uniform(18.0, 55.0)),
            shooter_alt_m=altitude,
            shooter_speed_mps=float(rng.uniform(230.0, 420.0)),
            target_alt_m=float(np.clip(altitude + rng.normal(0.0, 2_000.0), 1_000.0, 17_000.0)),
            target_speed_mps=float(rng.uniform(190.0, 360.0)),
            aspect=aspects[int(rng.integers(0, len(aspects)))],
            maneuver="hard_turn" if rng.random() < 0.35 else "none",
            countermeasure=countermeasures[int(rng.integers(0, len(countermeasures)))],
            jammer_burn_through_km=15.0 if rng.random() < 0.3 else 0.0,
            tick_s=0.5,
            guidance_reliability=float(rng.uniform(0.75, 1.0)),
            seeker_reliability=float(rng.uniform(0.75, 1.0)),
            datalink_reliability=float(rng.uniform(0.75, 1.0)),
            fuze_reliability=float(rng.uniform(0.80, 1.0)),
            warhead_effectiveness=float(rng.uniform(0.65, 0.95)),
        )
        rows.append(_run_case(case, seed + 100_000 + replicate))

    summaries = _summaries(rows)
    from bvr_marl_core.validation.studies import _save_figure, _write_csv

    _write_csv(output / "missile_launch_to_terminal.csv", rows)
    _write_csv(output / "missile_launch_to_terminal_summary.csv", summaries)

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    geometry = [item for item in summaries if item["family"] == "geometry"]
    axes[0, 0].bar(
        [item["case"] for item in geometry], [item["terminal_fraction"] for item in geometry]
    )
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].set_title("Terminal-event fraction by geometry")
    ew = [item for item in summaries if item["family"] == "ew"]
    axes[0, 1].bar([item["case"] for item in ew], [item["seduction_fraction"] for item in ew])
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].set_title("Countermeasure seduction fraction")
    dt_rows = [
        row for row in rows if row["family"] == "timestep" and row["miss_distance_m"] is not None
    ]
    for name in sorted({row["case"] for row in dt_rows}):
        values = [row["miss_distance_m"] for row in dt_rows if row["case"] == name]
        axes[1, 0].scatter([float(name.removeprefix("dt_"))] * len(values), values, label=name)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Macro timestep (s)")
    axes[1, 0].set_ylabel("CPA miss distance (m)")
    uncertainty = [row for row in rows if row["family"] == "uncertainty"]
    axes[1, 1].scatter(
        [row["range_km"] for row in uncertainty],
        [row["pk"] for row in uncertainty],
        c=[row["terminal_event"] for row in uncertainty],
        cmap="coolwarm",
    )
    axes[1, 1].set_xlabel("Launch range (km)")
    axes[1, 1].set_ylabel("Realized conditional $P_k$")
    axes[1, 1].set_title("Randomized parameter campaign")
    _save_figure(output / "missile_launch_to_terminal.png")

    uncertainty_summary = next(item for item in summaries if item["family"] == "uncertainty")
    timestep_medians = {
        item["case"]: item["median_miss_distance_m"]
        for item in summaries
        if item["family"] == "timestep"
    }
    timestep_values = [value for value in timestep_medians.values() if value is not None]
    return {
        "shots": len(rows),
        "replicates_per_structured_case": 4,
        "randomized_parameter_shots": 24,
        "families": sorted({row["family"] for row in rows}),
        "summaries": summaries,
        "timestep_median_miss_distance_m": timestep_medians,
        "maximum_timestep_median_miss_distance_spread_m": max(timestep_values)
        - min(timestep_values),
        "completed_lifecycle_fraction": sum(row["lifecycle_complete"] for row in rows) / len(rows),
        "randomized_kill_fraction": uncertainty_summary["kill_fraction"],
        "randomized_kill_ci95": [
            uncertainty_summary["kill_ci95_low"],
            uncertainty_summary["kill_ci95_high"],
        ],
        "scope": (
            "production launch-to-terminal dynamics; pairwise coverage of altitude, speed, "
            "aspect, seeker acquisition, support interruption, countermeasures, jamming, "
            "macro timestep, and randomized synthetic parameters"
        ),
        "interpretation": "internal model validation, not weapon-specific external calibration",
    }
