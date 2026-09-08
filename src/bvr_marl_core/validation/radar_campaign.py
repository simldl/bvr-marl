"""Seeded radar acquisition, confirmation, clutter, notch, and timestep studies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.radar.obs.observation import DEFAULT_NOTCH_VELOCITY_MPS
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator


class _MapLimits:
    bottom_lat = -5.0
    top_lat = 5.0
    left_lon = -5.0
    right_lon = 5.0
    min_alt = 0.0
    max_alt = 20_000.0


def _fighter(lat: float, lon: float, yaw: float, speed: float, group: str):
    return Eurofighter(
        Position(lat, lon, 9_000.0),
        yaw_deg=yaw,
        speed_mps=speed,
        group=group,
        map_limits=_MapLimits(),
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )


def _set_false_alarm_rate(radar, rate_hz: float) -> None:
    radar.false_alarm_rate = float(rate_hz)
    radar.param_policy.search_false_alarm_rate = float(rate_hz)


def _truth_for_report(simulator: Simulator, report) -> int | None:
    return simulator._sensor_report_truth_associations.get((report.source_id, report.report_id))


def _run_acquisition(range_km: float, dt_s: float, seed: int, duration_s: float = 60.0):
    simulator = Simulator(tick_secs=dt_s, random_seed=seed)
    simulator.record_traces = False
    blue = _fighter(0.0, 0.0, 90.0, 250.0, "blue")
    red = _fighter(0.0, range_km / 111.0, 270.0, 250.0, "red")
    simulator.add_unit(blue)
    simulator.add_unit(red)
    _set_false_alarm_rate(blue.radar, 0.0)
    blue.radar.obsgen.notch_velocity_mps = 0.0

    first_detection_s = None
    first_confirmed_s = None
    detections = 0
    max_tracks = 0
    ticks = int(round(duration_s / dt_s))
    for tick in range(ticks):
        simulator.do_tick()
        now_s = (tick + 1) * dt_s
        own_reports = [
            report
            for report in blue.radar.cached_detections or ()
            if _truth_for_report(simulator, report) == red.id
        ]
        detections += len(own_reports)
        if own_reports and first_detection_s is None:
            first_detection_s = now_s
        tracks = blue.radar.cached_tracks or ()
        max_tracks = max(max_tracks, len(tracks))
        if first_confirmed_s is None:
            confirmed = [
                track
                for track in tracks
                if track.engageable
                and simulator.evaluator_truth_id_for_contact(blue.id, track.track_id) == red.id
            ]
            if confirmed:
                first_confirmed_s = now_s
    return {
        "range_km": range_km,
        "tick_s": dt_s,
        "seed": seed,
        "duration_s": duration_s,
        "detected": first_detection_s is not None,
        "confirmed": first_confirmed_s is not None,
        "first_detection_s": first_detection_s,
        "first_confirmation_s": first_confirmed_s,
        "detection_count": detections,
        "max_receiver_tracks": max_tracks,
    }


def _run_false_alarm(seed: int, rate_hz: float = 2.0, duration_s: float = 60.0):
    simulator = Simulator(tick_secs=1.0, random_seed=seed)
    simulator.record_traces = False
    blue = _fighter(0.0, 0.0, 90.0, 250.0, "blue")
    simulator.add_unit(blue)
    _set_false_alarm_rate(blue.radar, rate_hz)
    blue.radar.obsgen.notch_velocity_mps = 0.0

    reports = 0
    confirmed_track_ticks = 0
    max_tentative = 0
    max_confirmed = 0
    max_total = 0
    for _ in range(int(duration_s)):
        simulator.do_tick()
        reports += sum(
            bool(report.get("is_false_alarm", False))
            for report in blue.radar.cached_detections or ()
        )
        tracks = blue.radar.cached_tracks or ()
        tentative = sum(track.lifecycle.value == "tentative" for track in tracks)
        confirmed = sum(track.engageable for track in tracks)
        confirmed_track_ticks += confirmed
        max_tentative = max(max_tentative, tentative)
        max_confirmed = max(max_confirmed, confirmed)
        max_total = max(max_total, len(tracks))
    return {
        "seed": seed,
        "duration_s": duration_s,
        "false_alarm_rate_hz": rate_hz,
        "false_reports": reports,
        "expected_false_reports": rate_hz * duration_s,
        "max_tentative_tracks": max_tentative,
        "max_confirmed_tracks": max_confirmed,
        "confirmed_track_seconds": confirmed_track_ticks,
        "max_total_tracks": max_total,
    }


def _notch_rows(seed: int) -> list[dict[str, Any]]:
    blue = _fighter(0.0, 0.0, 90.0, 0.0, "blue")
    radar = blue.radar
    radar.obsgen.notch_velocity_mps = DEFAULT_NOTCH_VELOCITY_MPS
    radar.obsgen.np_rng = np.random.default_rng(seed)
    target = _fighter(0.0, 50.0 / 111.0, 270.0, 0.0, "red")
    own_velocity = np.zeros(3, dtype=float)
    rows = []
    samples = 4_000
    for dt_s in (0.25, 0.5, 1.0):
        for radial_mps in (0.0, 12.5, 25.0, 37.5, 50.0, 100.0):
            target.speed = radial_mps
            target.yaw_deg = 90.0
            detected = 0
            for _ in range(samples):
                detected += bool(
                    radar.obsgen.generate(
                        blue.position,
                        [target],
                        yaw_deg=90.0,
                        pitch_deg=0.0,
                        own_group="blue",
                        own_id=blue.id,
                        own_velocity=own_velocity,
                        dwell_time_s=dt_s,
                    )
                )
            factor = radar.obsgen._notch_detection_factor(
                50_000.0,
                0.0,
                0.0,
                50_000.0,
                target,
                own_velocity,
            )
            rows.append(
                {
                    "tick_s": dt_s,
                    "radial_velocity_mps": radial_mps,
                    "notch_factor": factor,
                    "samples": samples,
                    "detected": detected,
                    "detection_fraction": detected / samples,
                }
            )
    return rows


def _distribution_summary(rows: list[dict[str, Any]], value_key: str) -> dict[str, Any]:
    values = [float(row[value_key]) for row in rows if row[value_key] is not None]
    return {
        "trials": len(rows),
        "observed": len(values),
        "observation_fraction": len(values) / len(rows),
        "median_s": float(np.median(values)) if values else None,
        "p05_s": float(np.quantile(values, 0.05)) if values else None,
        "p95_s": float(np.quantile(values, 0.95)) if values else None,
    }


def radar_operational_validation(output: Path, seed: int) -> dict[str, Any]:
    """Exercise production scan, report, track, clutter, and notch paths."""
    acquisition_rows = [
        _run_acquisition(range_km, dt_s, seed + index * 1_000 + replicate)
        for index, (range_km, dt_s) in enumerate(
            [(range_km, dt_s) for range_km in (30.0, 80.0, 140.0) for dt_s in (0.25, 0.5, 1.0)]
        )
        for replicate in range(30)
    ]
    false_alarm_rows = [_run_false_alarm(seed + 50_000 + replicate) for replicate in range(40)]
    notch_rows = _notch_rows(seed + 70_000)

    from bvr_marl_core.validation.studies import _save_figure, _write_csv

    _write_csv(output / "radar_acquisition_confirmation.csv", acquisition_rows)
    _write_csv(output / "radar_false_tracks.csv", false_alarm_rows)
    _write_csv(output / "radar_notch_timestep.csv", notch_rows)

    summaries: dict[str, Any] = {}
    for range_km in (30.0, 80.0, 140.0):
        for dt_s in (0.25, 0.5, 1.0):
            subset = [
                row
                for row in acquisition_rows
                if row["range_km"] == range_km and row["tick_s"] == dt_s
            ]
            summaries[f"range={range_km:g}km,dt={dt_s:g}s"] = {
                "detection": _distribution_summary(subset, "first_detection_s"),
                "confirmation": _distribution_summary(subset, "first_confirmation_s"),
            }

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for range_km in (30.0, 80.0, 140.0):
        medians = [
            summaries[f"range={range_km:g}km,dt={dt_s:g}s"]["detection"]["median_s"]
            for dt_s in (0.25, 0.5, 1.0)
        ]
        axes[0].plot((0.25, 0.5, 1.0), medians, marker="o", label=f"{range_km:g} km")
    axes[0].set_xlabel("Timestep (s)")
    axes[0].set_ylabel("Median first detection (s)")
    axes[0].set_title("Acquisition timestep sensitivity")
    axes[0].legend()
    for dt_s in (0.25, 0.5, 1.0):
        subset = [row for row in notch_rows if row["tick_s"] == dt_s]
        axes[1].plot(
            [row["radial_velocity_mps"] for row in subset],
            [row["detection_fraction"] for row in subset],
            marker="o",
            label=f"dt={dt_s:g}s",
        )
    axes[1].axvline(DEFAULT_NOTCH_VELOCITY_MPS, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Absolute radial velocity (m/s)")
    axes[1].set_ylabel("Detection fraction per dwell")
    axes[1].set_title("Doppler notch response")
    axes[1].legend()
    axes[2].hist(
        [row["max_confirmed_tracks"] for row in false_alarm_rows], bins=np.arange(-0.5, 6.5)
    )
    axes[2].set_xlabel("Maximum confirmed false tracks")
    axes[2].set_ylabel("Trials")
    axes[2].set_title("60 s clutter-only trials")
    _save_figure(output / "radar_operational_validation.png")

    false_reports = np.asarray([row["false_reports"] for row in false_alarm_rows], dtype=float)
    expected_false = false_alarm_rows[0]["expected_false_reports"]
    max_confirmed = max(row["max_confirmed_tracks"] for row in false_alarm_rows)
    confirmation_seconds = sum(row["confirmed_track_seconds"] for row in false_alarm_rows)
    detection_timestep_spread_s = 0.0
    confirmation_timestep_spread_s = 0.0
    minimum_confirmation_fraction = 1.0
    for range_km in (30.0, 80.0, 140.0):
        range_summaries = [
            summaries[f"range={range_km:g}km,dt={dt_s:g}s"] for dt_s in (0.25, 0.5, 1.0)
        ]
        detection_medians = [item["detection"]["median_s"] for item in range_summaries]
        confirmation_medians = [item["confirmation"]["median_s"] for item in range_summaries]
        detection_timestep_spread_s = max(
            detection_timestep_spread_s, max(detection_medians) - min(detection_medians)
        )
        minimum_confirmation_fraction = min(
            minimum_confirmation_fraction,
            *(item["confirmation"]["observation_fraction"] for item in range_summaries),
        )
        if all(value is not None for value in confirmation_medians):
            confirmation_timestep_spread_s = max(
                confirmation_timestep_spread_s,
                max(confirmation_medians) - min(confirmation_medians),
            )
    return {
        "acquisition_trials": len(acquisition_rows),
        "replicates_per_range_timestep": 30,
        "acquisition": summaries,
        "false_track_trials": len(false_alarm_rows),
        "false_alarm_rate_hz": false_alarm_rows[0]["false_alarm_rate_hz"],
        "mean_false_reports": float(np.mean(false_reports)),
        "expected_false_reports": expected_false,
        "false_report_mean_standard_error": float(
            np.std(false_reports, ddof=1) / math.sqrt(len(false_reports))
        ),
        "maximum_confirmed_false_tracks": max_confirmed,
        "confirmed_false_track_seconds": confirmation_seconds,
        "maximum_detection_median_timestep_spread_s": detection_timestep_spread_s,
        "maximum_confirmation_median_timestep_spread_s": confirmation_timestep_spread_s,
        "minimum_confirmation_fraction": minimum_confirmation_fraction,
        "notch_samples_per_cell": notch_rows[0]["samples"],
        "notch_zero_radial_detection_fraction": {
            str(dt_s): next(
                row["detection_fraction"]
                for row in notch_rows
                if row["tick_s"] == dt_s and row["radial_velocity_mps"] == 0.0
            )
            for dt_s in (0.25, 0.5, 1.0)
        },
        "scope": (
            "production sector scan, hazard-based detections, report freezing, anonymous tracking, "
            "Poisson false alarms, and Doppler-notch response"
        ),
        "interpretation": "synthetic internal model validation; no named-radar calibration",
    }
