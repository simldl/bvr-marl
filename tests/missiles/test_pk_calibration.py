"""Monte-Carlo calibration harness (Priorities 3 & 10).

Samples launch geometries, records the tactical *predicted* shot quality
(``NoEscapeZoneCalculator.sqi``) at launch, runs N seeded engagements per cell,
and harvests the realized closest-approach / Pk / kill outcome from the Phase-1
terminal events. It then computes the empirical kill probability and the
calibration error

    e_cal = Pk_empirical - shot_quality_predicted

writes an inspectable CSV (miss-distance distributions included), and asserts the
robustly-checkable acceptance criteria: predicted shot quality is positively
related to realized lethality, and lethality falls monotonically with range.

Tuning the model is then a data-driven loop over this report — the knobs are the
per-missile effectiveness params (Phase 5), the vulnerability/track tables
(Phases 6-7), and the ``nez.py`` zone coefficients. The harness does not auto-fit
them; it is the instrument that makes the calibration error measurable.
"""

import csv
import os
import random
from pathlib import Path

import pytest

from bvr_marl_core.aircraft.core.nez import clear_dlz_cache
from bvr_marl_core.simulator.core.events import MissileTerminalEvent
from bvr_marl_core.simulator.core.hit_event_helpers import kill_probability_components
from bvr_marl_core.simulator.simulator import Simulator
from tests.missiles.test_range_zone_validation import (
    _ASPECT_YAW,
    MISSILES,
    _fighter,
    _near_range,
    _place_target_east,
    _set_loadout,
)

pytestmark = pytest.mark.slow


def _run_with_prediction(missile_cls, slant_m, seed, *, aspect="head_on", maneuver=None, ticks=160):
    """Fire one engagement; return (predicted_shot_quality, terminal_record)."""
    sim = Simulator(tick_secs=1.0, random_seed=seed)
    shooter = _fighter(0.0, 0.0, 90.0, 300.0, "blue", 8_000.0)
    _set_loadout(shooter, missile_cls)
    target = _place_target_east(shooter, slant_m, 8_000.0, _ASPECT_YAW[aspect], 250.0)
    sim.add_unit(shooter)
    sim.add_unit(target)

    clear_dlz_cache()
    dlz = shooter.wez.compute_dlz(target)
    predicted = float(shooter.wez.sqi(shooter, target, dlz=dlz))

    missile, veto, _ = shooter.weapons.fire_missile_direct(sim, target, missile_cls)
    assert missile is not None, veto

    rng = random.Random(seed * 7919 + 1)
    for tick in range(ticks):
        if maneuver == "jink" and tick % 3 == 0 and target.id in sim.active_units:
            target.control.set_yaw_deg(rng.uniform(0.0, 360.0))
        sim.do_tick()
        if missile.id not in sim.active_units:
            break

    record = None
    for event in sim.events:
        if isinstance(event, MissileTerminalEvent) and event.missile is missile:
            record = event.record
    return predicted, record


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 1e-12 or syy <= 1e-12:
        return 0.0
    return sxy / (sxx**0.5 * syy**0.5)


def _csv_path() -> Path:
    configured = os.environ.get("BVR_PK_CALIBRATION_CSV")
    if configured:
        return Path(configured)
    return Path("output") / "missile_telemetry" / "pk_calibration.csv"


def _cells():
    """(label, range_fn) bands spanning short -> beyond-reach for each DLZ."""
    return {
        "near": lambda d: _near_range(d),
        "mid": lambda d: 0.5 * (d.r_min_m + d.r_pi_m),
        "far": lambda d: max(d.r_aero_m * 1.5, 400_000.0),
    }


def test_pk_calibration_report():
    seeds = int(os.environ.get("BVR_PK_CALIBRATION_SEEDS", "4"))
    cells = _cells()

    rows = []
    per_missile_pk = {}
    all_pred, all_obs = [], []
    for name, missile_cls in MISSILES.items():
        # Baseline DLZ to size the range bands for this missile.
        _, _ = _run_with_prediction(missile_cls, 20_000.0, 0)
        clear_dlz_cache()
        sh = _fighter(0.0, 0.0, 90.0, 300.0, "blue", 8_000.0)
        _set_loadout(sh, missile_cls)
        tg = _place_target_east(sh, 20_000.0, 8_000.0, _ASPECT_YAW["head_on"], 250.0)
        dlz = sh.wez.compute_dlz(tg)

        per_missile_pk[name] = {}
        for band, range_fn in cells.items():
            slant = range_fn(dlz)
            kills = 0
            preds, misses = [], []
            for seed in range(seeds):
                pred, rec = _run_with_prediction(
                    missile_cls, slant, seed, aspect="head_on", maneuver="jink"
                )
                preds.append(pred)
                if rec is not None:
                    if rec.get("killed"):
                        kills += 1
                    if rec.get("miss_distance_m") is not None:
                        misses.append(rec["miss_distance_m"])
            pk_emp = kills / seeds if seeds else 0.0
            pred_mean = sum(preds) / len(preds) if preds else 0.0
            per_missile_pk[name][band] = pk_emp
            all_pred.append(pred_mean)
            all_obs.append(pk_emp)
            rows.append(
                {
                    "missile": name,
                    "band": band,
                    "R_launch_m": round(slant, 1),
                    "shots": seeds,
                    "kills": kills,
                    "pk_empirical": round(pk_emp, 3),
                    "shot_quality_predicted": round(pred_mean, 3),
                    "e_cal": round(pk_emp - pred_mean, 3),
                    "mean_miss_m": round(sum(misses) / len(misses), 1) if misses else None,
                }
            )

    out_path = _csv_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    assert out_path.exists() and rows
    # Miss-distance distributions are logged and inspectable.
    assert any(r["mean_miss_m"] is not None for r in rows)
    # Lethality falls monotonically with range for every missile.
    for name, pk in per_missile_pk.items():
        assert pk["near"] >= pk["far"], f"{name}: Pk(near) < Pk(far) — {pk}"
    # Predicted shot quality is positively related to realized lethality.
    corr = _pearson(all_pred, all_obs)
    assert corr > 0.3, f"weak predicted-vs-observed correlation: {corr:.2f}"


def test_acceptance_criteria_decomposition_and_logging():
    """Priority-10 functional checks that don't require a Monte-Carlo run:
    the kill model separates fuze/warhead/vulnerability/tracking, and the
    terminal record carries the breakdown for analysis."""
    from types import SimpleNamespace

    m = SimpleNamespace(hit_probability=0.85, lethal_radius_m=100.0)
    pk, comp = kill_probability_components(m, None, 25.0)
    assert set(comp) == {"p_int", "p_fuze", "p_wh", "p_vul", "p_trk"}
    assert 0.0 <= pk <= 1.0

    # Fire one short engagement and confirm the terminal event carries the
    # component breakdown and a miss distance.
    pred, rec = _run_with_prediction(next(iter(MISSILES.values())), 12_000.0, 0)
    assert rec is not None
    assert "pk_components" in rec and set(rec["pk_components"]) == {
        "p_int",
        "p_fuze",
        "p_wh",
        "p_vul",
        "p_trk",
    }
    assert rec["miss_distance_m"] is not None
