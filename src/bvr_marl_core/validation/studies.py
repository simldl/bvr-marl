"""Deterministic studies built directly on production simulator models.

These studies verify internal behavior and sensitivity. They do not constitute
calibration against classified or otherwise unavailable real-world reference data.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import platform
import time
import tracemalloc
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.domain.information import TrackLifecycle, TrackSnapshot
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.physics.aircraft import AircraftPhysics
from bvr_marl_core.radar.core.lut import DetectionLUT
from bvr_marl_core.radar.core.utils import _doppler, has_effective_earth_line_of_sight
from bvr_marl_core.radar.tracking.filter.filters import (
    ConstantVelocityKFFilter,
    create_ct_filter,
    create_imm_cv_ct_filter,
)
from bvr_marl_core.radar.tracking.helpers.track_manager import TRACK_DELETION_MISSED_UPDATES
from bvr_marl_core.radar.tracking.tracker import TrackerManager
from bvr_marl_core.rl.environment.spaces.observation.enemy_info_builder import EnemyInfoBuilder
from bvr_marl_core.simulator.core.effectiveness.kill_model import KillProbabilityModel
from bvr_marl_core.simulator.core.effectiveness.terminal_track import TerminalTrackQualityModel
from bvr_marl_core.simulator.core.experiment_metadata import build_experiment_metadata
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.core.units import FlyingUnit
from bvr_marl_core.simulator.simulator import Simulator
from bvr_marl_core.validation.missile_campaign import missile_launch_to_terminal
from bvr_marl_core.validation.radar_campaign import radar_operational_validation

Study = Callable[[Path, int], dict[str, Any]]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def physics_envelope(output: Path, seed: int) -> dict[str, Any]:
    """Evaluate production instantaneous/sustained turn envelopes."""
    del seed  # deterministic grid study
    model = AircraftPhysics(AircraftPhysics.Params(mass_kg=18_000.0, reference_area_m2=50.0))
    speeds = np.arange(100.0, 651.0, 10.0)
    altitudes = (0.0, 5_000.0, 10_000.0, 15_000.0)
    rows: list[dict[str, Any]] = []
    peaks: dict[str, dict[str, float]] = {}
    plt.figure(figsize=(8.0, 4.8))
    for altitude in altitudes:
        sustained = []
        instantaneous = []
        for speed in speeds:
            inst = model.compute_instantaneous_turn_rate(float(speed), altitude)
            sust = model.compute_sustained_turn_rate(float(speed), altitude, throttle=1.0)
            ps_level = model.force_model.compute_specific_energy_rate(
                float(speed), altitude, 1.0, 1.0
            )
            rows.append(
                {
                    "altitude_m": altitude,
                    "speed_mps": float(speed),
                    "instantaneous_turn_deg_s": inst,
                    "sustained_turn_deg_s": sust,
                    "level_specific_energy_rate_m_s": ps_level,
                }
            )
            instantaneous.append(inst)
            sustained.append(sust)
        peak_index = int(np.argmax(sustained))
        peaks[str(int(altitude))] = {
            "peak_sustained_turn_deg_s": float(sustained[peak_index]),
            "speed_at_peak_mps": float(speeds[peak_index]),
            "peak_instantaneous_turn_deg_s": float(max(instantaneous)),
        }
        plt.plot(speeds, sustained, label=f"{altitude / 1000:.0f} km")
    plt.xlabel("True airspeed (m/s)")
    plt.ylabel("Sustained turn rate (deg/s)")
    plt.title("Production aircraft turn envelope")
    plt.grid(alpha=0.25)
    plt.legend(title="Altitude")
    _save_figure(output / "physics_turn_envelope.png")
    _write_csv(output / "physics_turn_envelope.csv", rows)
    return {"grid_points": len(rows), "peaks_by_altitude_m": peaks}


def radar_sensitivity(output: Path, seed: int) -> dict[str, Any]:
    """Evaluate Pd surfaces, horizon boundaries, Doppler, and Monte Carlo error."""
    rng = np.random.default_rng(seed)
    lut = DetectionLUT(
        freq_hz=10e9,
        tx_power_w=15_000.0,
        gain=10 ** (40.0 / 10.0),
        max_range_m=500_000.0,
        snr_threshold_db=10.0,
        max_rcs=100.0,
        rcs_bins=240,
        dist_bins=500,
        processing_gain_db=25.0,
    )
    ranges_km = np.arange(10.0, 501.0, 5.0)
    rcs_values = (1e-4, 1e-2, 1.0, 10.0, 100.0, 1_000.0)
    rows: list[dict[str, Any]] = []
    pd50: dict[str, float | None] = {}
    plt.figure(figsize=(8.0, 4.8))
    for rcs in rcs_values:
        probabilities = np.array(
            [lut.get_probability(float(distance * 1000.0), rcs) for distance in ranges_km]
        )
        for distance, probability in zip(ranges_km, probabilities, strict=True):
            rows.append({"range_km": distance, "rcs_m2": rcs, "pd": probability})
        valid = ranges_km[probabilities >= 0.5]
        pd50[str(rcs)] = float(valid[-1]) if len(valid) else None
        plt.plot(ranges_km, probabilities, label=f"{rcs:g} m²")
    plt.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Slant range (km)")
    plt.ylabel("Probability of detection per evaluation")
    plt.title("Radar LUT sensitivity")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(title="Effective RCS")
    _save_figure(output / "radar_pd_sensitivity.png")
    _write_csv(output / "radar_pd_sensitivity.csv", rows)

    horizon_rows = []
    for sensor_alt in (100.0, 1_000.0, 10_000.0):
        for target_alt in (100.0, 1_000.0, 10_000.0):
            radius = 6_371_000.0 * 4.0 / 3.0
            boundary = math.sqrt(2 * radius * sensor_alt + sensor_alt**2) + math.sqrt(
                2 * radius * target_alt + target_alt**2
            )
            horizon_rows.append(
                {
                    "sensor_altitude_m": sensor_alt,
                    "target_altitude_m": target_alt,
                    "horizon_range_km": boundary / 1000.0,
                    "inside_gate": has_effective_earth_line_of_sight(
                        boundary - 1.0, sensor_alt, target_alt
                    ),
                    "outside_gate": has_effective_earth_line_of_sight(
                        boundary + 1.0, sensor_alt, target_alt
                    ),
                }
            )
    _write_csv(output / "radar_horizon.csv", horizon_rows)

    class Target:
        velocity = np.array([250.0, -50.0, 20.0])

    geometries = [(0.0, 0.0), (90.0, 0.0), (45.0, 20.0), (0.0, 90.0)]
    radar_velocity = np.array([100.0, -50.0, 0.0])
    doppler_errors = []
    for azimuth, elevation in geometries:
        ar, er = math.radians(azimuth), math.radians(elevation)
        los = np.array([math.cos(er) * math.sin(ar), math.cos(er) * math.cos(ar), math.sin(er)])
        expected = 2.0 * 10e9 * float(np.dot(Target.velocity - radar_velocity, los)) / 299_792_458.0
        actual = _doppler(Target(), azimuth, elevation, 10e9, radar_velocity)
        doppler_errors.append(abs(actual - expected))

    monte_carlo_n = 20_000
    probe_pd = lut.get_probability(150_000.0, 1.0)
    observed = float(np.mean(rng.random(monte_carlo_n) < probe_pd))
    se = math.sqrt(probe_pd * (1.0 - probe_pd) / monte_carlo_n)
    return {
        "pd50_range_km_by_rcs_m2": pd50,
        "high_rcs_clamp_range_dependent": lut.get_probability(50_000.0, 1_000.0)
        > lut.get_probability(450_000.0, 1_000.0),
        "horizon_cases": len(horizon_rows),
        "max_doppler_error_hz": max(doppler_errors),
        "monte_carlo_probe": {
            "range_km": 150.0,
            "rcs_m2": 1.0,
            "samples": monte_carlo_n,
            "expected_pd": probe_pd,
            "observed_pd": observed,
            "three_sigma_half_width": 3.0 * se,
        },
    }


# Filter initialization transient. Velocity is seeded at zero against a target
# flying at roughly 241 m/s, so the first samples are dominated by that seed and
# not by steady-state tracking accuracy. Metrics are reported both pooled (all
# steps, retained for continuity) and restricted to the converged regime.
_TRACKING_WARMUP_STEPS = 10
# Measurement outage for the ``dropout`` scenario, as a 0-based step index.
# Step ``k`` is plotted at ``(k + 1) * dt`` seconds.
_DROPOUT_FIRST_MISSED_STEP = 25
# The production tracker deletes a track once it reaches
# TRACK_DELETION_MISSED_UPDATES missed updates, so the oldest coasted state it
# can ever export is one update short of that. The validated dropout scenario
# withholds exactly that many updates: any longer outage describes a track the
# simulator has already dropped, and its error is not tracker behaviour.
_DROPOUT_MISSED_STEPS = TRACK_DELETION_MISSED_UPDATES - 1
# Unbounded filter-only reference. Retained because the coast response of the
# constant-velocity model is worth showing, but it leaves the tracker's
# operating envelope at _DROPOUT_MISSED_STEPS and is reported and drawn as a
# reference curve, never as tracker validation evidence.
_DROPOUT_REFERENCE_MISSED_STEPS = 10
# Steps scored as "recovery" once updates resume, before the converged regime.
_DROPOUT_RECOVERY_STEPS = 5


def _nees(error: np.ndarray, covariance: np.ndarray) -> float:
    """Normalized estimation error squared, e^T P^-1 e.

    ``solve`` keeps a singular covariance visible as a failure instead of
    silently returning the minimum-norm answer a pseudo-inverse would give, and
    it avoids an SVD in the innermost Monte Carlo loop.
    """
    try:
        return float(error @ np.linalg.solve(covariance, error))
    except np.linalg.LinAlgError:
        return float("nan")


def _rms_standard_error(squared_errors: np.ndarray) -> np.ndarray:
    """Monte Carlo standard error of a per-step RMS taken over independent runs."""
    runs = squared_errors.shape[0]
    mean_square = np.mean(squared_errors, axis=0)
    # Delta method on rms = sqrt(mean_square): se(rms) = se(mean_square) / (2*rms).
    mean_square_se = np.std(squared_errors, axis=0, ddof=1) / math.sqrt(runs)
    return np.where(mean_square > 0.0, mean_square_se / (2.0 * np.sqrt(mean_square)), 0.0)


def tracking_consistency(output: Path, seed: int) -> dict[str, Any]:
    """Monte Carlo CV tracking error, NIS, NEES, maneuver, and dropout study.

    The ``dropout`` scenario withholds the longest run of measurement updates a
    track actually survives: the track manager deletes at
    ``TRACK_DELETION_MISSED_UPDATES`` missed updates, so the oldest coasted state
    the tracker can export is one update short of that. Its peak is therefore the
    largest coast error the simulator can ever produce.

    ``dropout_beyond_deletion`` keeps coasting well past that threshold. Because
    this study drives the filter directly and never constructs a
    ``TrackerManager``, nothing stops it leaving the tracker's operating
    envelope, and its error keeps ramping for a track the simulator would have
    dropped. It is retained because the unbounded coast response of the
    constant-velocity model is worth showing, but it is reported and drawn as a
    filter-only reference and must not be read as tracker validation.

    Scenarios share common random numbers -- run ``r`` draws the same
    measurement-noise stream in every scenario, and the draw happens on every
    step whether or not the update is applied -- so the three curves are paired
    and remain comparable across the outage instead of diverging into
    independent noise realizations.
    """
    runs, steps, dt = 300, 60, 1.0
    measurement_std = np.array([120.0, 120.0, 180.0])
    # "dropout" withholds the longest outage a track actually survives. The
    # reference scenario keeps coasting past deletion; it is reported and drawn
    # separately and is not tracker validation evidence.
    outage_steps = {
        "straight": 0,
        "turn": 0,
        "dropout": _DROPOUT_MISSED_STEPS,
        "dropout_beyond_deletion": _DROPOUT_REFERENCE_MISSED_STEPS,
    }
    scenarios = tuple(outage_steps)
    # One stream per run, reused across scenarios (common random numbers).
    run_seeds = np.random.SeedSequence(seed).spawn(runs)

    step_index = np.arange(steps)
    times = (step_index + 1) * dt

    def scenario_masks(name: str) -> tuple[np.ndarray, np.ndarray]:
        count = outage_steps[name]
        coast = np.zeros(steps, dtype=bool)
        coast[_DROPOUT_FIRST_MISSED_STEP : _DROPOUT_FIRST_MISSED_STEP + count] = True
        recovery = np.zeros(steps, dtype=bool)
        if count:
            start = _DROPOUT_FIRST_MISSED_STEP + count
            recovery[start : start + _DROPOUT_RECOVERY_STEPS] = True
        return coast, recovery

    dropout_coast, _ = scenario_masks("dropout")
    outage_window_s = (float(times[dropout_coast][0]), float(times[dropout_coast][-1]))
    # One more missed update than the retained coast and the track is deleted.
    deletion_boundary_s = float(
        times[_DROPOUT_FIRST_MISSED_STEP + TRACK_DELETION_MISSED_UPDATES - 1]
    )
    reference_coast, _ = scenario_masks("dropout_beyond_deletion")
    reference_end_s = float(times[reference_coast][-1])

    rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, float]] = {}
    plt.figure(figsize=(8.0, 4.8))
    axes = plt.gca()
    axes.axvspan(
        outage_window_s[0] - 0.5,
        outage_window_s[1] + 0.5,
        color="0.85",
        zorder=0,
        label=f"outage ({_DROPOUT_MISSED_STEPS:.0f} s: longest coast the tracker retains)",
    )
    axes.axvspan(
        outage_window_s[1] + 0.5,
        reference_end_s + 0.5,
        color="#f3d9d4",
        zorder=0,
        label="beyond deletion: track no longer exists",
    )
    axes.axvline(
        deletion_boundary_s - 0.5,
        color="#a33",
        linestyle="--",
        linewidth=1.1,
        zorder=4,
        label=f"deletion threshold ({TRACK_DELETION_MISSED_UPDATES} missed updates)",
    )
    axes.axvspan(
        times[0] - 0.5,
        times[_TRACKING_WARMUP_STEPS - 1] + 0.5,
        color="0.93",
        zorder=0,
        label="filter initialization transient",
    )

    for scenario in scenarios:
        coast_steps, recovery_steps = scenario_masks(scenario)
        recovery_start = _DROPOUT_FIRST_MISSED_STEP + outage_steps[scenario]
        step_errors = np.zeros((runs, steps))
        velocity_errors = np.zeros((runs, steps))
        nis_by_step = np.full((runs, steps), np.nan)
        nees_by_step = np.zeros((runs, steps))
        for run in range(runs):
            rng = np.random.default_rng(run_seeds[run])
            position = np.zeros(3)
            velocity = np.array([240.0, 20.0, 2.0])
            first_measurement = position + rng.normal(0.0, measurement_std)
            kf = ConstantVelocityKFFilter(
                first_measurement,
                dt,
                process_noise_std=8.0,
                R_diag=tuple(measurement_std**2),
            )
            for step in range(steps):
                if scenario == "turn" and step >= 20:
                    angle = math.radians(2.0 * dt)
                    velocity[:2] = np.array(
                        [
                            math.cos(angle) * velocity[0] - math.sin(angle) * velocity[1],
                            math.sin(angle) * velocity[0] + math.cos(angle) * velocity[1],
                        ]
                    )
                position += velocity * dt
                kf.predict(dt)
                # Drawn unconditionally so a withheld update cannot desynchronize
                # this run's noise stream from the same run in another scenario.
                measurement = position + rng.normal(0.0, measurement_std)
                if not coast_steps[step]:
                    kf.update(measurement)
                    nis_by_step[run, step] = float(kf.get_last_update_stats()["nis"])
                estimate = kf.get_state()
                error = estimate - np.concatenate((position, velocity))
                step_errors[run, step] = np.linalg.norm(error[:3])
                velocity_errors[run, step] = np.linalg.norm(error[3:])
                nees_by_step[run, step] = _nees(error, kf.get_covariance())

        squared_position = step_errors**2
        rms_by_step = np.sqrt(np.mean(squared_position, axis=0))
        rms_se_by_step = _rms_standard_error(squared_position)

        converged = (step_index >= _TRACKING_WARMUP_STEPS) & ~coast_steps & ~recovery_steps
        missed_count = np.where(coast_steps, step_index - _DROPOUT_FIRST_MISSED_STEP + 1, 0)
        regimes = np.where(
            coast_steps & (missed_count >= TRACK_DELETION_MISSED_UPDATES),
            "coast_beyond_deletion",
            np.where(
                coast_steps,
                "coast",
                np.where(
                    recovery_steps,
                    "recovery",
                    np.where(step_index < _TRACKING_WARMUP_STEPS, "initialization", "converged"),
                ),
            ),
        )
        for index in range(steps):
            rows.append(
                {
                    "scenario": scenario,
                    "time_s": float(times[index]),
                    "position_rms_m": float(rms_by_step[index]),
                    "position_rms_standard_error_m": float(rms_se_by_step[index]),
                    "regime": str(regimes[index]),
                    "measurement_applied": not bool(coast_steps[index]),
                }
            )

        entry = {
            "position_rms_m": float(np.sqrt(np.mean(squared_position))),
            "final_position_rms_m": float(rms_by_step[-1]),
            "velocity_rms_mps": float(np.sqrt(np.mean(velocity_errors**2))),
            "mean_nis": float(np.nanmean(nis_by_step)),
            "mean_nees": float(np.nanmean(nees_by_step)),
            "converged_position_rms_m": float(np.sqrt(np.mean(squared_position[:, converged]))),
            "converged_position_rms_standard_error_m": float(
                math.sqrt(float(np.mean(rms_se_by_step[converged] ** 2)) / int(converged.sum()))
            ),
            "converged_velocity_rms_mps": float(
                np.sqrt(np.mean(velocity_errors[:, converged] ** 2))
            ),
            "converged_mean_nis": float(np.nanmean(nis_by_step[:, converged])),
            "converged_mean_nees": float(np.nanmean(nees_by_step[:, converged])),
        }
        if outage_steps[scenario]:
            coast_rms = rms_by_step[coast_steps]
            converged_rms = entry["converged_position_rms_m"]
            after = rms_by_step[recovery_start:]
            within_tolerance = np.flatnonzero(after <= 1.10 * converged_rms)
            entry.update(
                {
                    "coast_peak_position_rms_m": float(coast_rms[-1]),
                    "coast_growth_m_per_s": float(
                        (coast_rms[-1] - rms_by_step[_DROPOUT_FIRST_MISSED_STEP - 1])
                        / (outage_steps[scenario] * dt)
                    ),
                    "coast_peak_over_converged_ratio": float(coast_rms[-1] / converged_rms),
                    "reacquisition_recovery_s": (
                        float((within_tolerance[0] + 1) * dt)
                        if within_tolerance.size
                        else float("nan")
                    ),
                }
            )
        if scenario == "dropout":
            # The largest coast error the tracker can ever export: one update
            # short of the deletion threshold.
            entry["max_reachable_coast_rms_m"] = entry["coast_peak_position_rms_m"]
        if scenario == "dropout_beyond_deletion":
            entry["tracker_reachable"] = False
            entry["scope"] = (
                "filter-only reference; the track manager deletes this track at "
                f"{TRACK_DELETION_MISSED_UPDATES} missed updates, so this curve is not "
                "tracker behaviour and is not validation evidence"
            )
        aggregate[scenario] = entry

        reference = scenario == "dropout_beyond_deletion"
        line = axes.plot(
            times,
            rms_by_step,
            label="dropout (filter-only, past deletion)" if reference else scenario,
            linestyle="--" if reference else "-",
            color="0.45" if reference else None,
            linewidth=1.3 if reference else 1.6,
            zorder=3,
        )[0]
        axes.fill_between(
            times,
            rms_by_step - rms_se_by_step,
            rms_by_step + rms_se_by_step,
            color=line.get_color(),
            alpha=0.25,
            linewidth=0,
            zorder=2,
        )

    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Position RMS error (m)")
    axes.set_title(
        f"CV tracker Monte Carlo consistency ({runs} paired runs per case, band = ±1 SE)"
    )
    axes.grid(alpha=0.25, zorder=1)
    reachable_peak = aggregate["dropout"]["max_reachable_coast_rms_m"]
    reference_peak = aggregate["dropout_beyond_deletion"]["coast_peak_position_rms_m"]
    # Headroom so the legend cannot sit on top of the initialization transient.
    axes.set_ylim(top=reference_peak * 1.30)
    axes.annotate(
        f"max coast error the tracker\ncan export: {reachable_peak:.0f} m",
        xy=(outage_window_s[1], reachable_peak),
        xytext=(outage_window_s[1] - 13.0, reachable_peak + 0.20 * reference_peak),
        textcoords="data",
        fontsize=8,
        va="center",
        ha="left",
        arrowprops={"arrowstyle": "->", "color": "0.35", "linewidth": 0.9},
    )
    axes.legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.9)
    _save_figure(output / "tracking_consistency.png")
    _write_csv(output / "tracking_consistency.csv", rows)
    return {
        "runs_per_scenario": runs,
        "steps_per_run": steps,
        "measurement_std_m": measurement_std.tolist(),
        "expected_nis_mean_for_3d_measurement": 3.0,
        "expected_nees_mean_for_6d_state": 6.0,
        "common_random_numbers_across_scenarios": True,
        "initialization_transient_steps": _TRACKING_WARMUP_STEPS,
        "dropout_outage_window_s": list(outage_window_s),
        "track_deletion_missed_updates": TRACK_DELETION_MISSED_UPDATES,
        "track_deletion_boundary_s": deletion_boundary_s,
        "validated_scenarios": ["straight", "turn", "dropout"],
        "reference_only_scenarios": ["dropout_beyond_deletion"],
        "regime_definitions": {
            "initialization": f"t <= {_TRACKING_WARMUP_STEPS * dt:.0f} s, velocity seeded at zero",
            "coast": "measurement updates withheld, track still retained by the track manager",
            "coast_beyond_deletion": (
                "measurement updates withheld past the deletion threshold; the track manager "
                "has already removed this track, so these samples are filter-only reference"
            ),
            "recovery": (
                f"first {_DROPOUT_RECOVERY_STEPS} steps after updates resume in an outage scenario"
            ),
            "converged": "all remaining steps; the basis of the converged_* metrics",
        },
        "scenarios": aggregate,
    }


def tracking_model_comparison(output: Path, seed: int) -> dict[str, Any]:
    """Compare CV, Cartesian CT, and CV/CT IMM on shared noisy trajectories.

    ``dropout`` withholds the longest run of updates a track survives before the
    track manager deletes it, so the comparison is scored over a regime the
    simulator can actually reach. ``dropout_beyond_deletion`` continues coasting
    past that threshold and is reported separately as a filter-only reference;
    the declared acceptance gate is evaluated on the reachable case only.
    """
    rng = np.random.default_rng(seed)
    runs, steps, dt = 100, 60, 1.0
    measurement_std = np.array([120.0, 120.0, 180.0])
    measurement_covariance = np.diag(measurement_std**2)
    outage_steps = {
        "straight": 0,
        "turn": 0,
        "dropout": _DROPOUT_MISSED_STEPS,
        "dropout_beyond_deletion": _DROPOUT_REFERENCE_MISSED_STEPS,
    }
    scenarios = tuple(outage_steps)
    validated_scenarios = ("straight", "turn", "dropout")
    squared_errors: dict[tuple[str, str], list[float]] = {
        (model, scenario): [] for model in ("cv", "ct", "imm_cv_ct") for scenario in scenarios
    }
    mode_probabilities: list[float] = []

    for scenario in scenarios:
        for _ in range(runs):
            position = np.zeros(3)
            velocity = np.array([240.0, 20.0, 2.0])
            first = position + rng.normal(0.0, measurement_std)
            models = {
                "cv": ConstantVelocityKFFilter(
                    first,
                    dt,
                    process_noise_std=8.0,
                    R_diag=tuple(measurement_std**2),
                ),
                "ct": create_ct_filter(first, dt, measurement_covariance),
                "imm_cv_ct": create_imm_cv_ct_filter(
                    first,
                    dt,
                    measurement_covariance=measurement_covariance,
                ),
            }
            for step in range(steps):
                if scenario == "turn" and step >= 20:
                    angle = math.radians(2.0 * dt)
                    cosine, sine = math.cos(angle), math.sin(angle)
                    velocity[:2] = np.array(
                        [
                            cosine * velocity[0] - sine * velocity[1],
                            sine * velocity[0] + cosine * velocity[1],
                        ]
                    )
                position += velocity * dt
                measurement = position + rng.normal(0.0, measurement_std)
                missed = (
                    _DROPOUT_FIRST_MISSED_STEP
                    <= step
                    < _DROPOUT_FIRST_MISSED_STEP + outage_steps[scenario]
                )
                for name, model in models.items():
                    model.predict(dt)
                    if not missed:
                        model.update(measurement)
                    error = model.get_state()[:3] - position
                    squared_errors[(name, scenario)].append(float(error @ error))
                # Sampled every step, not just at the end of the run: the
                # declared gate is that mode probabilities stay finite and
                # normalized throughout, including across the coast.
                mode_probabilities.extend(models["imm_cv_ct"].mode_probabilities.tolist())

    rows = [
        {
            "model": model,
            "scenario": scenario,
            "position_rms_m": math.sqrt(float(np.mean(squared_errors[(model, scenario)]))),
            "tracker_reachable": scenario in validated_scenarios,
        }
        for scenario in scenarios
        for model in ("cv", "ct", "imm_cv_ct")
    ]
    _write_csv(output / "tracking_model_comparison.csv", rows)

    def rms(model: str, scenario: str) -> float:
        return next(
            row["position_rms_m"]
            for row in rows
            if row["model"] == model and row["scenario"] == scenario
        )

    plt.figure(figsize=(7.6, 4.5))
    x = np.arange(len(scenarios), dtype=float)
    width = 0.25
    for index, model in enumerate(("cv", "ct", "imm_cv_ct")):
        values = [rms(model, scenario) for scenario in scenarios]
        bars = plt.bar(x + (index - 1) * width, values, width=width, label=model)
        # Hatch the reference group so it cannot be mistaken for a scored case.
        for scenario, bar in zip(scenarios, bars, strict=True):
            if scenario not in validated_scenarios:
                bar.set_hatch("//")
                bar.set_alpha(0.55)
    plt.xticks(
        x,
        [
            "straight",
            "turn",
            f"dropout\n({_DROPOUT_MISSED_STEPS} s, retained)",
            f"beyond deletion\n({_DROPOUT_REFERENCE_MISSED_STEPS} s, filter only)",
        ],
        fontsize=8,
    )
    plt.ylabel("Position RMS error (m)")
    plt.title("Tracker motion-model comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    _save_figure(output / "tracking_model_comparison.png")
    by_model = {
        model: {scenario: rms(model, scenario) for scenario in validated_scenarios}
        for model in ("cv", "ct", "imm_cv_ct")
    }
    reference_by_model = {
        model: rms(model, "dropout_beyond_deletion") for model in ("cv", "ct", "imm_cv_ct")
    }
    return {
        "runs_per_scenario": runs,
        "steps_per_run": steps,
        "measurement_std_m": measurement_std.tolist(),
        "models": by_model,
        "dropout_beyond_deletion_reference_rms_m": reference_by_model,
        "track_deletion_missed_updates": TRACK_DELETION_MISSED_UPDATES,
        "gate_scope": (
            "acceptance is evaluated on tracker-reachable scenarios only; the "
            "beyond-deletion column is a filter-only reference"
        ),
        "mode_probabilities_finite_and_normalized": bool(
            np.all(np.isfinite(mode_probabilities))
            and all(
                np.isclose(sum(mode_probabilities[index : index + 2]), 1.0)
                for index in range(0, len(mode_probabilities), 2)
            )
        ),
        "scope": "synthetic internal CV/CT/IMM comparison; CV remains the runtime default",
    }


def association_crossing(output: Path, seed: int) -> dict[str, Any]:
    """Score anonymous-track identity through noisy two-target crossings."""
    rng = np.random.default_rng(seed)
    runs, steps, dt = 100, 41, 1.0
    measurement_std = np.array([80.0, 80.0, 100.0])
    lateral_separations_m = (100.0, 300.0, 1_000.0, 5_000.0, 10_000.0)
    reference = Position(0.0, 0.0, 0.0)
    rows: list[dict[str, Any]] = []

    def cluster(position: np.ndarray, report_id: int) -> dict[str, Any]:
        east, north, up = position
        distance = float(np.linalg.norm(position))
        return {
            "az": math.degrees(math.atan2(east, north)) % 360.0,
            "el": math.degrees(math.asin(up / distance)),
            "d": distance,
            "dop": 0.0,
            "source_pos": reference,
            "report_ids": (report_id,),
            "source_ids": (3,),
            "n_obs": 1,
            "covariance_cartesian": np.diag(measurement_std**2),
            "measurement_ref": reference,
        }

    for separation in lateral_separations_m:
        swap_events = 0
        identity_transitions = 0
        fragmented_truth_trajectories = 0
        assigned_track_ids = 0
        scored_updates = 0
        squared_errors: list[float] = []
        for run in range(runs):
            manager = TrackerManager(assoc_dist=250.0)
            histories: list[list[int]] = [[], []]
            for step in range(steps):
                crossing_axis = -4_000.0 + 200.0 * step
                truths = (
                    np.array([crossing_axis, -separation / 2.0, 8_000.0]),
                    np.array([-crossing_axis, separation / 2.0, 8_000.0]),
                )
                reports = [
                    cluster(
                        truth + rng.normal(0.0, measurement_std),
                        run * steps * 2 + step * 2 + truth_index,
                    )
                    for truth_index, truth in enumerate(truths)
                ]
                rng.shuffle(reports)
                tracks = manager.update_tracks(reports, dt, reference)
                if len(tracks) < 2:
                    continue
                track_positions = [np.asarray(track.state, dtype=float)[:3] for track in tracks]
                assignment = min(
                    itertools.permutations(range(len(tracks)), 2),
                    key=lambda indices: sum(
                        np.linalg.norm(track_positions[indices[index]] - truths[index])
                        for index in range(2)
                    ),
                )
                scored_updates += 1
                for truth_index, track_index in enumerate(assignment):
                    track_id = int(tracks[track_index].track_id)
                    history = histories[truth_index]
                    if history:
                        identity_transitions += 1
                        swap_events += track_id != history[-1]
                    history.append(track_id)
                    squared_errors.append(
                        float(np.sum((track_positions[track_index] - truths[truth_index]) ** 2))
                    )
            for history in histories:
                unique_ids = len(set(history))
                assigned_track_ids += unique_ids
                fragmented_truth_trajectories += unique_ids > 1
        truth_trajectories = runs * 2
        row = {
            "lateral_separation_m": separation,
            "runs": runs,
            "scored_update_fraction": scored_updates / (runs * steps),
            "identity_swap_rate": swap_events / max(identity_transitions, 1),
            "fragmented_trajectory_rate": fragmented_truth_trajectories / truth_trajectories,
            "mean_track_ids_per_truth_trajectory": assigned_track_ids / truth_trajectories,
            "position_rms_m": math.sqrt(float(np.mean(squared_errors))),
        }
        rows.append(row)

    _write_csv(output / "association_crossing.csv", rows)
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(
        [row["lateral_separation_m"] for row in rows],
        [100.0 * row["identity_swap_rate"] for row in rows],
        marker="o",
        label="identity transitions",
    )
    plt.plot(
        [row["lateral_separation_m"] for row in rows],
        [100.0 * row["fragmented_trajectory_rate"] for row in rows],
        marker="s",
        label="fragmented trajectories",
    )
    plt.xlabel("Lateral crossing separation (m)")
    plt.ylabel("Rate (%)")
    plt.title("Anonymous-track association through crossings")
    plt.grid(alpha=0.25)
    plt.legend()
    _save_figure(output / "association_crossing.png")
    return {
        "runs_per_separation": runs,
        "steps_per_run": steps,
        "measurement_std_enu_m": measurement_std.tolist(),
        "rows": rows,
        "truth_use": "post-update scoring only; reports and tracker association are anonymous",
    }


def datalink_age_sensitivity(output: Path, seed: int) -> dict[str, Any]:
    """Measure CV coast error and covariance growth for delayed maneuver tracks."""
    del seed
    speed = 250.0
    turn_rate = math.radians(3.0)
    delays = np.arange(0.0, 16.0, 1.0)
    rows: list[dict[str, Any]] = []
    for delay in delays:
        state = (0.0, 0.0, 8_000.0, speed, 0.0, 0.0)
        covariance = np.diag([100.0**2] * 3 + [10.0**2] * 3)
        snapshot = TrackSnapshot(
            track_id=1,
            state_time_s=0.0,
            state=state,
            covariance=tuple(tuple(row) for row in covariance),
            confidence=0.9,
            lifecycle=TrackLifecycle.CONFIRMED,
            source_ids=(7,),
        )
        coasted = snapshot.propagated(float(delay), maneuver_accel_std_mps2=10.0)
        if delay == 0.0:
            true_position = np.array(state[:3])
        else:
            true_position = np.array(
                [
                    speed / turn_rate * math.sin(turn_rate * delay),
                    speed / turn_rate * (1.0 - math.cos(turn_rate * delay)),
                    8_000.0,
                ]
            )
        estimate = np.asarray(coasted.state[:3])
        coast_covariance = np.asarray(coasted.covariance)
        rows.append(
            {
                "track_age_s": float(delay),
                "maneuver_position_error_m": float(np.linalg.norm(estimate - true_position)),
                "horizontal_sigma_m": float(
                    math.sqrt(coast_covariance[0, 0] + coast_covariance[1, 1])
                ),
                "reported_age_s": coasted.age_s,
            }
        )
    _write_csv(output / "datalink_age_sensitivity.csv", rows)
    plt.figure(figsize=(8.0, 4.8))
    plt.plot(delays, [row["maneuver_position_error_m"] for row in rows], label="turn error")
    plt.plot(delays, [row["horizontal_sigma_m"] for row in rows], label="reported 2-D σ")
    plt.xlabel("Track age (s)")
    plt.ylabel("Distance (m)")
    plt.title("Datalink coast sensitivity to track age")
    plt.grid(alpha=0.25)
    plt.legend()
    _save_figure(output / "datalink_age_sensitivity.png")
    return {
        "turn_rate_deg_s": 3.0,
        "speed_mps": speed,
        "maximum_track_age_s": float(delays[-1]),
        "maximum_position_error_m": rows[-1]["maneuver_position_error_m"],
        "maximum_horizontal_sigma_m": rows[-1]["horizontal_sigma_m"],
        "age_is_monotonic": all(
            rows[index]["reported_age_s"] <= rows[index + 1]["reported_age_s"]
            for index in range(len(rows) - 1)
        ),
        "error_within_reported_horizontal_sigma": all(
            row["maneuver_position_error_m"] <= row["horizontal_sigma_m"] for row in rows
        ),
    }


def missile_effectiveness(output: Path, seed: int) -> dict[str, Any]:
    """Evaluate decomposed terminal Pk versus miss distance and track quality."""
    rng = np.random.default_rng(seed)

    class Provider:
        def __init__(self, age_s: float, locked: bool):
            self.last_confirmed_track_age_s = age_s
            self.locked = locked

        def has_fresh_track(self) -> bool:
            return self.locked

        def has_coastable_track(self) -> bool:
            return False

    distances = np.arange(0.0, 101.0, 2.0)
    rows: list[dict[str, Any]] = []
    model = KillProbabilityModel(
        track=TerminalTrackQualityModel(lambda_age=0.12, lambda_cov=1.0, no_lock_factor=0.4)
    )
    cases = ((0.0, 0.0, True), (5.0, 0.3, True), (10.0, 0.7, False))
    plt.figure(figsize=(8.0, 4.8))
    summaries = {}
    for age_s, uncertainty, locked in cases:
        missile = SimpleNamespace(
            guidance_reliability=0.95,
            fuze_reliability=0.98,
            warhead_effectiveness=0.85,
            lethal_radius_m=25.0,
            terminal_track_uncertainty=uncertainty,
            target_provider=Provider(age_s, locked),
            radar=None,
        )
        probabilities = []
        for distance in distances:
            pk, components = model.compute(missile, None, float(distance))
            probabilities.append(pk)
            rows.append(
                {
                    "miss_distance_m": float(distance),
                    "track_age_s": age_s,
                    "track_uncertainty": uncertainty,
                    "terminal_lock": locked,
                    "pk": pk,
                    **components,
                }
            )
        label = f"age={age_s:g}s, uncertainty={uncertainty:g}, lock={locked}"
        plt.plot(distances, probabilities, label=label)
        sample_count = 50_000
        pk_zero = probabilities[0]
        observed = float(np.mean(rng.random(sample_count) < pk_zero))
        summaries[label] = {
            "pk_at_zero_m": pk_zero,
            "pk_at_50_m": probabilities[25],
            "monte_carlo_samples": sample_count,
            "observed_kill_rate_at_zero_m": observed,
        }
    plt.xlabel("Closest-approach miss distance (m)")
    plt.ylabel("Conditional kill probability")
    plt.title("Missile terminal-effectiveness sensitivity")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    _save_figure(output / "missile_effectiveness.png")
    _write_csv(output / "missile_effectiveness.csv", rows)
    return {"cases": summaries, "external_calibration": "not performed"}


def terminal_parameter_randomization(output: Path, seed: int) -> dict[str, Any]:
    """Randomize terminal-model inputs across seeds and achieved-CPA regimes."""

    class Provider:
        def __init__(self, age_s: float, locked: bool):
            self.last_confirmed_track_age_s = age_s
            self.locked = locked

        def has_fresh_track(self) -> bool:
            return self.locked

        def has_coastable_track(self) -> bool:
            return False

    model = KillProbabilityModel(
        track=TerminalTrackQualityModel(lambda_age=0.12, lambda_cov=1.0, no_lock_factor=0.4)
    )
    regimes = {
        "tight": (5.0, 5.0),
        "nominal": (20.0, 12.0),
        "wide": (45.0, 20.0),
    }
    seed_replicates, samples_per_replicate = 30, 300
    rows: list[dict[str, Any]] = []
    for regime, (miss_mean, miss_std) in regimes.items():
        for replicate in range(seed_replicates):
            run_seed = seed + 10_000 * list(regimes).index(regime) + replicate
            rng = np.random.default_rng(run_seed)
            probabilities = []
            outcomes = []
            miss_distances = []
            for _ in range(samples_per_replicate):
                miss_distance = max(0.0, float(rng.normal(miss_mean, miss_std)))
                provider = Provider(float(rng.uniform(0.0, 10.0)), bool(rng.random() >= 0.2))
                missile = SimpleNamespace(
                    guidance_reliability=float(rng.uniform(0.85, 0.99)),
                    fuze_reliability=float(rng.uniform(0.90, 0.995)),
                    warhead_effectiveness=float(rng.uniform(0.65, 0.95)),
                    lethal_radius_m=float(rng.uniform(15.0, 35.0)),
                    terminal_track_uncertainty=float(rng.uniform(0.0, 0.8)),
                    target_provider=provider,
                    radar=None,
                )
                probability, _ = model.compute(missile, None, miss_distance)
                probabilities.append(probability)
                outcomes.append(float(rng.random() < probability))
                miss_distances.append(miss_distance)
            rows.append(
                {
                    "cpa_regime": regime,
                    "seed": run_seed,
                    "samples": samples_per_replicate,
                    "mean_miss_distance_m": float(np.mean(miss_distances)),
                    "mean_conditional_pk": float(np.mean(probabilities)),
                    "realized_kill_fraction": float(np.mean(outcomes)),
                }
            )

    _write_csv(output / "terminal_parameter_randomization.csv", rows)
    summaries: dict[str, dict[str, float]] = {}
    plt.figure(figsize=(7.2, 4.5))
    for index, regime in enumerate(regimes):
        values = np.asarray(
            [row["realized_kill_fraction"] for row in rows if row["cpa_regime"] == regime]
        )
        mean = float(np.mean(values))
        low, high = (float(value) for value in np.percentile(values, (5.0, 95.0)))
        summaries[regime] = {
            "mean_realized_kill_fraction": mean,
            "seed_5th_percentile": low,
            "seed_95th_percentile": high,
            "mean_conditional_pk": float(
                np.mean([row["mean_conditional_pk"] for row in rows if row["cpa_regime"] == regime])
            ),
        }
        plt.errorbar(
            index,
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            fmt="o",
            capsize=5,
            color="#264653",
        )
    plt.xticks(range(len(regimes)), list(regimes))
    plt.xlabel("Achieved closest-approach regime")
    plt.ylabel("Realized conditional kill fraction")
    plt.title("Terminal-model parameter and seed randomization")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25, axis="y")
    _save_figure(output / "terminal_parameter_randomization.png")
    return {
        "seed_replicates": seed_replicates,
        "samples_per_replicate": samples_per_replicate,
        "randomized_inputs": [
            "miss_distance",
            "guidance_reliability",
            "fuze_reliability",
            "warhead_effectiveness",
            "lethal_radius",
            "track_age",
            "track_uncertainty",
            "terminal_lock",
        ],
        "regimes": summaries,
        "scope": "conditional on achieved closest approach; no flyout or tactical validation",
    }


def tactical_invariance(output: Path, seed: int) -> dict[str, Any]:
    """Check roster insertion-order invariance and straight-flight dt sensitivity."""

    def trajectory(order: tuple[int, ...], dt: float) -> dict[int, tuple[float, float, float]]:
        units = {}
        for unit_id in order:
            unit = FlyingUnit(
                f"unit-{unit_id}",
                Position(48.0 + 0.01 * unit_id, 11.0, 8_000.0),
                20.0 * unit_id,
                220.0 + unit_id,
            )
            unit.id = unit_id
            units[unit_id] = unit
        sim = Simulator(tick_secs=dt, random_seed=seed)
        sim.record_traces = False
        sim.reset_sim(units)
        for _ in range(round(20.0 / dt)):
            sim.do_tick()
        return {
            unit_id: (unit.position.lat, unit.position.lon, unit.position.alt)
            for unit_id, unit in sim.active_units.items()
        }

    forward = trajectory((1, 2, 3, 4), 1.0)
    reverse = trajectory((4, 3, 2, 1), 1.0)
    fine = trajectory((1, 2, 3, 4), 0.25)
    order_delta = max(
        np.linalg.norm(np.asarray(forward[unit_id]) - np.asarray(reverse[unit_id]))
        for unit_id in forward
    )
    dt_delta = max(
        np.linalg.norm(np.asarray(forward[unit_id]) - np.asarray(fine[unit_id]))
        for unit_id in forward
    )
    rows = [
        {
            "unit_id": unit_id,
            "latitude_deg": forward[unit_id][0],
            "longitude_deg": forward[unit_id][1],
            "altitude_m": forward[unit_id][2],
        }
        for unit_id in sorted(forward)
    ]
    _write_csv(output / "tactical_invariance.csv", rows)
    return {
        "unit_count": len(forward),
        "duration_s": 20.0,
        "maximum_insertion_order_delta_coordinate_units": float(order_delta),
        "maximum_dt_1_to_0_25_delta_coordinate_units": float(dt_delta),
        "insertion_order_identical": forward == reverse,
    }


def information_mode_invariance(output: Path, seed: int) -> dict[str, Any]:
    """Perturb hidden evaluator attributes while holding operational tracks fixed."""
    del seed  # deterministic contract study

    class Config:
        em_slots = 1
        ef_slots = 1

        def __init__(self, mode: str):
            self.information_mode = mode

    ownship = SimpleNamespace(
        id=1,
        yaw_deg=0.0,
        position=SimpleNamespace(lat=48.0, lon=11.0, alt=8_000.0),
        target=None,
    )
    ownship.sensor = SimpleNamespace(
        sensor_tracks=[],
        get_nez_features=lambda simulator, selected_target_id: {},
        get_locked_targets=lambda: set(),
        bda_confirmed=set(),
    )
    simulator = SimpleNamespace(active_units={1: ownship})

    def observation(mode: str, *, hidden_support_flag: bool) -> np.ndarray:
        evaluator_target = SimpleNamespace(
            id=9001,
            is_missile=False,
            is_support_asset=hidden_support_flag,
            is_non_engageable=False,
        )
        simulator.active_units[41] = evaluator_target
        ownship.sensor.sensor_tracks = [
            TrackSnapshot(
                track_id=41,
                state_time_s=1.0,
                state=(50_000.0, 5_000.0, 1_000.0, -200.0, 20.0, 0.0),
                covariance=tuple(tuple(row) for row in np.eye(6)),
                confidence=0.5,
                lifecycle=TrackLifecycle.CONFIRMED,
                classification_probabilities=(0.8, 0.1, 0.02, 0.01, 0.07),
                engageable=True,
            )
        ]
        missiles, fighters = EnemyInfoBuilder(simulator, Config(mode)).build(1)
        return np.concatenate((missiles.ravel(), fighters.ravel()))

    rows = []
    for mode in ("sensor_limited", "oracle"):
        baseline = observation(mode, hidden_support_flag=False)
        perturbed = observation(mode, hidden_support_flag=True)
        difference = np.abs(perturbed - baseline)
        rows.append(
            {
                "information_mode": mode,
                "hidden_truth_linf_delta": float(np.max(difference)),
                "hidden_truth_l2_delta": float(np.linalg.norm(difference)),
                "changed_features": int(np.count_nonzero(difference)),
            }
        )

    _write_csv(output / "information_mode_invariance.csv", rows)
    plt.figure(figsize=(6.5, 4.2))
    plt.bar(
        [row["information_mode"].replace("_", " ") for row in rows],
        [row["hidden_truth_linf_delta"] for row in rows],
        color=("#2a9d8f", "#e76f51"),
    )
    plt.ylabel("Maximum observation change")
    plt.title("Hidden-truth perturbation invariance")
    plt.grid(alpha=0.25, axis="y")
    _save_figure(output / "information_mode_invariance.png")

    by_mode = {row["information_mode"]: row for row in rows}
    return {
        "modes": by_mode,
        "sensor_limited_invariant": by_mode["sensor_limited"]["hidden_truth_linf_delta"] == 0.0,
        "oracle_control_responded": by_mode["oracle"]["hidden_truth_linf_delta"] > 0.0,
        "scope": "enemy observation builder; fixed operational track, perturbed evaluator support flag",
    }


MAX_RUNTIME_SCALING_EXPONENT = 2.0
"""Acceptance bound on the fitted runtime-scaling exponent.

The invariant worth asserting is **"scaling is no worse than quadratic"**. Radar
detection and track association are pairwise over N aircraft, so O(N^2) is the
inherent cost of the modelled physics; an exponent above 2.0 means something has
become genuinely super-quadratic (an accidental nested scan, an O(N^3) association
pass) and is a real regression.

This was 1.7, which is NOT such a bound -- it sat inside the measurement's own noise.
Measured over 18 consecutive runs of the tactical study on unchanged code: mean 1.664,
std 0.103, range [1.42, 1.85], with 30-50% of runs exceeding 1.7 depending on machine
load. The estimator is a 4-point log-log fit over medians of only ``repeats=3`` timings
of ``timed_ticks=20`` each, so its standard error is ~0.1 by construction; contention
from anything else on the box shifts it further. A threshold below the mean of that
distribution makes the test a coin flip, which trains people to ignore it.

Taking the fastest repeat instead of the median (contention only ever ADDS time) helps
only marginally -- measured mean 1.613, std 0.089 -- so it does not rescue a 1.7 bound
and is not worth changing a reported paper quantity for. Raise sampling here if a
tighter bound is ever genuinely needed: the standard error falls as 1/sqrt(repeats *
timed_ticks).
"""


def performance_scaling(output: Path, seed: int) -> dict[str, Any]:
    """Measure full Simulator.do_tick throughput versus active roster size."""
    counts = (2, 8, 32, 128)
    repeats, ticks = 5, 100
    rows = []
    for count in counts:
        rates = []
        for repeat in range(repeats):
            sim = Simulator(tick_secs=1.0, random_seed=seed + repeat)
            sim.record_traces = False
            for index in range(count):
                sim.add_unit(
                    FlyingUnit(
                        f"benchmark-{index}",
                        Position(48.0 + index * 1e-4, 11.0, 8_000.0),
                        float(index % 360),
                        250.0,
                    )
                )
            for _ in range(10):
                sim.do_tick()
            start = time.perf_counter()
            for _ in range(ticks):
                sim.do_tick()
            elapsed = time.perf_counter() - start
            rates.append(ticks / elapsed)
        rows.append(
            {
                "active_units": count,
                "median_ticks_per_second": float(np.median(rates)),
                "minimum_ticks_per_second": float(min(rates)),
                "maximum_ticks_per_second": float(max(rates)),
                "repeats": repeats,
                "ticks_per_repeat": ticks,
            }
        )
    _write_csv(output / "performance_scaling.csv", rows)
    plt.figure(figsize=(7.0, 4.5))
    plt.loglog(
        [row["active_units"] for row in rows],
        [row["median_ticks_per_second"] for row in rows],
        marker="o",
    )
    plt.xlabel("Active flying units")
    plt.ylabel("Simulator ticks per second")
    plt.title("Measured simulator roster scaling")
    plt.grid(alpha=0.25, which="both")
    _save_figure(output / "performance_scaling.png")
    log_count = np.log(np.asarray(counts, dtype=float))
    log_time = np.log(1.0 / np.asarray([row["median_ticks_per_second"] for row in rows]))
    exponent = float(np.polyfit(log_count, log_time, 1)[0])
    return {
        "rows": rows,
        "fitted_runtime_exponent": exponent,
        "scope": "FlyingUnit roster and complete do_tick orchestration; no radar or missiles",
    }


def tactical_performance_scaling(output: Path, seed: int) -> dict[str, Any]:
    """Measure full aircraft/sensor/weapon tick scaling and Python peak memory."""
    counts = (2, 4, 8, 16)
    repeats, warmup_ticks, timed_ticks = 3, 5, 20
    map_limits = SimpleNamespace(
        left_lon=-5.0,
        right_lon=5.0,
        bottom_lat=-5.0,
        top_lat=5.0,
        min_alt=0.0,
        max_alt=20_000.0,
    )

    def make_simulator(count: int, run_seed: int) -> Simulator:
        simulator = Simulator(tick_secs=1.0, random_seed=run_seed)
        simulator.record_traces = False
        for index in range(count):
            side = index % 2
            lane = index // 2
            position = Position(
                -0.2 + 0.01 * lane if side == 0 else 0.2 - 0.01 * lane,
                -0.03 + 0.02 * (lane % 4),
                8_000.0 + 100.0 * (lane % 3),
            )
            aircraft = Eurofighter(
                position,
                0.0 if side == 0 else 180.0,
                250.0,
                "blue" if side == 0 else "red",
                map_limits,
                0.0,
                20_000.0,
            )
            simulator.add_unit(aircraft)
        return simulator

    def launch_weapons(simulator: Simulator) -> None:
        blue = sorted(
            (u for u in simulator.active_units.values() if getattr(u, "group", None) == "blue"),
            key=lambda unit: unit.id,
        )
        red = sorted(
            (u for u in simulator.active_units.values() if getattr(u, "group", None) == "red"),
            key=lambda unit: unit.id,
        )
        for shooter, target in zip(blue, red):
            shooter.weapons.fire_missile_direct(simulator, target, AIM120_AMRAAM)

    rows: list[dict[str, Any]] = []
    for count in counts:
        rates: list[float] = []
        final_track_counts: list[int] = []
        tentative_track_counts: list[int] = []
        confirmed_track_counts: list[int] = []
        network_track_counts: list[int] = []
        network_picture_counts: list[int] = []
        final_report_counts: list[int] = []
        active_missile_counts: list[int] = []
        event_counts: list[int] = []
        for repeat in range(repeats):
            simulator = make_simulator(count, seed + repeat)
            for _ in range(warmup_ticks):
                simulator.do_tick()
            launch_weapons(simulator)
            start = time.perf_counter()
            for _ in range(timed_ticks):
                simulator.do_tick()
            elapsed = time.perf_counter() - start
            rates.append(timed_ticks / elapsed)
            sensors = [
                unit.sensor
                for unit in simulator.active_units.values()
                if getattr(unit, "sensor", None) is not None
            ]
            final_track_counts.append(sum(len(sensor.sensor_tracks) for sensor in sensors))
            lifecycles = [track.lifecycle for sensor in sensors for track in sensor.sensor_tracks]
            tentative_track_counts.append(
                sum(lifecycle is TrackLifecycle.TENTATIVE for lifecycle in lifecycles)
            )
            confirmed_track_counts.append(
                sum(lifecycle is not TrackLifecycle.TENTATIVE for lifecycle in lifecycles)
            )
            network_pictures = getattr(simulator, "network_pictures", {})
            network_track_counts.append(
                sum(len(picture.tracks) for picture in network_pictures.values())
            )
            network_picture_counts.append(len(network_pictures))
            final_report_counts.append(
                sum(len(sensor.radar.cached_detections or ()) for sensor in sensors)
            )
            active_missile_counts.append(
                sum(
                    bool(getattr(unit, "is_missile", False))
                    for unit in simulator.active_units.values()
                )
            )
            event_counts.append(len(simulator.events))

        tracemalloc.start()
        memory_simulator = make_simulator(count, seed)
        for _ in range(warmup_ticks):
            memory_simulator.do_tick()
        launch_weapons(memory_simulator)
        for _ in range(timed_ticks):
            memory_simulator.do_tick()
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(
            {
                "active_aircraft": count,
                "median_ticks_per_second": float(np.median(rates)),
                "minimum_ticks_per_second": float(min(rates)),
                "maximum_ticks_per_second": float(max(rates)),
                "median_final_tracks": float(np.median(final_track_counts)),
                "median_tentative_tracks": float(np.median(tentative_track_counts)),
                "median_confirmed_or_coasting_tracks": float(np.median(confirmed_track_counts)),
                "median_unique_network_tracks": float(np.median(network_track_counts)),
                "median_network_pictures": float(np.median(network_picture_counts)),
                "median_final_sensor_reports": float(np.median(final_report_counts)),
                "median_active_missiles": float(np.median(active_missile_counts)),
                "median_event_count": float(np.median(event_counts)),
                "peak_python_memory_mib": peak_bytes / (1024.0**2),
                "repeats": repeats,
                "timed_ticks_per_repeat": timed_ticks,
            }
        )

    _write_csv(output / "tactical_performance_scaling.csv", rows)
    figure, left_axis = plt.subplots(figsize=(7.4, 4.7))
    right_axis = left_axis.twinx()
    left_axis.loglog(
        [row["active_aircraft"] for row in rows],
        [row["median_ticks_per_second"] for row in rows],
        marker="o",
        color="#264653",
        label="throughput",
    )
    right_axis.plot(
        [row["active_aircraft"] for row in rows],
        [row["peak_python_memory_mib"] for row in rows],
        marker="s",
        color="#e76f51",
        label="peak memory",
    )
    left_axis.set_xlabel("Active radar-equipped aircraft")
    left_axis.set_ylabel("Simulator ticks per second", color="#264653")
    right_axis.set_ylabel("Peak traced Python memory (MiB)", color="#e76f51")
    left_axis.set_title("Full aircraft/radar/tracker scaling")
    left_axis.grid(alpha=0.25, which="both")
    figure.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    _save_figure(output / "tactical_performance_scaling.png")
    log_count = np.log(np.asarray(counts, dtype=float))
    log_time = np.log(1.0 / np.asarray([row["median_ticks_per_second"] for row in rows]))
    return {
        "rows": rows,
        "fitted_runtime_exponent": float(np.polyfit(log_count, log_time, 1)[0]),
        "scope": (
            "complete Simulator.do_tick with opposing Eurofighters, flight control, "
            "radar, tracking, launched missiles, terminal events, and logging"
        ),
        "memory_scope": "peak Python allocations measured by tracemalloc",
    }


STUDIES: dict[str, Study] = {
    "association": association_crossing,
    "datalink": datalink_age_sensitivity,
    "missile": missile_effectiveness,
    "missile_flyout": missile_launch_to_terminal,
    "missile_randomization": terminal_parameter_randomization,
    "performance": performance_scaling,
    "performance_tactical": tactical_performance_scaling,
    "information": information_mode_invariance,
    "physics": physics_envelope,
    "radar": radar_sensitivity,
    "radar_operational": radar_operational_validation,
    "tactical": tactical_invariance,
    "tracking": tracking_consistency,
    "tracking_models": tracking_model_comparison,
}


def _acceptance(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Evaluate explicit internal verification thresholds without claiming calibration."""
    checks: dict[str, dict[str, Any]] = {}
    if "missile_flyout" in results:
        missile = results["missile_flyout"]
        checks["missile_launch_to_terminal_completion"] = {
            "threshold": "all launched missiles reach detonation, miss expiry, or removal",
            "observed": missile["completed_lifecycle_fraction"],
            "passed": missile["completed_lifecycle_fraction"] == 1.0,
        }
        spread = missile["maximum_timestep_median_miss_distance_spread_m"]
        checks["missile_timestep_convergence"] = {
            "threshold": "maximum median CPA miss-distance spread <= 10 m",
            "observed": spread,
            "passed": spread <= 10.0,
        }
    if "radar_operational" in results:
        radar = results["radar_operational"]
        false_deviation = abs(radar["mean_false_reports"] - radar["expected_false_reports"])
        false_tolerance = 4.0 * radar["false_report_mean_standard_error"]
        checks["radar_detection_confirmation_timestep_invariance"] = {
            "threshold": "all trials confirm and median spreads <= 1 s",
            "observed": {
                "minimum_confirmation_fraction": radar["minimum_confirmation_fraction"],
                "detection_spread_s": radar["maximum_detection_median_timestep_spread_s"],
                "confirmation_spread_s": radar["maximum_confirmation_median_timestep_spread_s"],
            },
            "passed": radar["minimum_confirmation_fraction"] == 1.0
            and radar["maximum_detection_median_timestep_spread_s"] <= 1.0
            and radar["maximum_confirmation_median_timestep_spread_s"] <= 1.0,
        }
        checks["radar_false_track_control"] = {
            "threshold": "mean reports within four standard errors and no confirmed false track",
            "observed": {
                "mean_report_deviation": false_deviation,
                "four_standard_errors": false_tolerance,
                "maximum_confirmed_false_tracks": radar["maximum_confirmed_false_tracks"],
            },
            "passed": false_deviation <= false_tolerance
            and radar["maximum_confirmed_false_tracks"] == 0,
        }
        notch = radar["notch_zero_radial_detection_fraction"]
        checks["radar_notch_timestep_invariance"] = {
            "threshold": "zero detection at zero radial velocity for every timestep",
            "observed": notch,
            "passed": all(value == 0.0 for value in notch.values()),
        }
    if "association" in results:
        rows = results["association"]["rows"]
        # Two targets separated by less than roughly three lateral measurement
        # standard deviations are not statistically resolvable from position alone.
        # Keep the 100 m stress case in the evidence, but apply the nominal
        # separated-target acceptance gate from 300 m upward.
        resolvable_rows = [row for row in rows if row["lateral_separation_m"] >= 300.0]
        worst_fragmentation = max(row["fragmented_trajectory_rate"] for row in resolvable_rows)
        checks["association_fragmentation"] = {
            "threshold": "<= 0.10 for resolvable separation >= 300 m",
            "observed": worst_fragmentation,
            "passed": worst_fragmentation <= 0.10,
        }
    if "tracking_models" in results:
        models = results["tracking_models"]["models"]
        imm = models["imm_cv_ct"]
        cv = models["cv"]
        normalized = results["tracking_models"]["mode_probabilities_finite_and_normalized"]
        checks["ct_imm_motion_model_comparison"] = {
            "threshold": (
                "IMM turn RMS <= CV, straight RMS <= 1.15*CV, "
                "dropout RMS <= 1.25*CV, finite normalized mode probabilities"
            ),
            "observed": {
                "straight_ratio": imm["straight"] / cv["straight"],
                "turn_ratio": imm["turn"] / cv["turn"],
                "dropout_ratio": imm["dropout"] / cv["dropout"],
                "mode_probabilities_finite_and_normalized": normalized,
            },
            "passed": bool(
                imm["turn"] <= cv["turn"]
                and imm["straight"] <= 1.15 * cv["straight"]
                and imm["dropout"] <= 1.25 * cv["dropout"]
                and normalized
            ),
        }
    if "information" in results:
        observed = bool(results["information"]["sensor_limited_invariant"])
        checks["sensor_limited_truth_invariance"] = {
            "threshold": "true",
            "observed": observed,
            "passed": observed,
        }
    if "tactical" in results:
        tactical = results["tactical"]
        checks["tick_order_and_dt_convergence"] = {
            "threshold": "order invariant and dt coordinate delta <= 1e-4",
            "observed": tactical["maximum_dt_1_to_0_25_delta_coordinate_units"],
            "passed": bool(tactical["insertion_order_identical"])
            and tactical["maximum_dt_1_to_0_25_delta_coordinate_units"] <= 1e-4,
        }
    if "performance_tactical" in results:
        performance = results["performance_tactical"]
        rows = performance["rows"]
        eight_vs_eight = next(row for row in rows if row["active_aircraft"] == 16)
        checks["full_stack_8v8_throughput"] = {
            "threshold": (
                ">= 2 ticks/s, <= 16 confirmed/coasting tracks/receiver, "
                f"and fitted runtime exponent <= {MAX_RUNTIME_SCALING_EXPONENT:g}"
            ),
            "observed": {
                "ticks_per_second": eight_vs_eight["median_ticks_per_second"],
                "confirmed_tracks_per_receiver": eight_vs_eight[
                    "median_confirmed_or_coasting_tracks"
                ]
                / 16.0,
                "fitted_runtime_exponent": performance["fitted_runtime_exponent"],
            },
            "passed": eight_vs_eight["median_ticks_per_second"] >= 2.0
            and eight_vs_eight["median_confirmed_or_coasting_tracks"] / 16.0 <= 16.0
            and performance["fitted_runtime_exponent"] <= MAX_RUNTIME_SCALING_EXPONENT,
        }
    return checks


def run_studies(
    output: Path, names: Iterable[str] | None = None, seed: int = 20260717
) -> dict[str, Any]:
    """Run selected studies and write a reproducible manifest and summary."""
    # GUI tests and interactive applications may change the process-wide backend
    # after this module is imported. Reassert headless rendering for every run.
    plt.switch_backend("Agg")
    selected = list(names or STUDIES)
    unknown = sorted(set(selected) - STUDIES.keys())
    if unknown:
        raise ValueError(f"Unknown studies: {', '.join(unknown)}")
    output.mkdir(parents=True, exist_ok=True)
    results = {name: STUDIES[name](output, seed) for name in selected}
    manifest = {
        "seed": seed,
        "study_names": selected,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scope": "internal verification and sensitivity; not external empirical calibration",
        "experiment_metadata": build_experiment_metadata({"seed": seed, "studies": selected}),
        "results": results,
        "acceptance": _acceptance(results),
    }
    (output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
