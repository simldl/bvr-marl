"""Range / NEZ / DLZ validation campaign (Priorities 2 & 8 of the weapon-
effectiveness plan).

This is an *offline* correctness harness, not a tuning experiment. For each
Fox-3 missile type it places a shooter and target at a controlled launch
geometry, fires (bypassing the radar-lock gate so the geometry is exactly as
set), runs the engagement to completion, and harvests the Phase-1
``MissileTerminalEvent`` (closest-approach miss, Pk, kill/no-kill). It then
checks the qualitative correctness properties the plan requires:

  * a shot inside the practical launch band reaches the target and can kill;
  * a shot well beyond the predicted aerodynamic range essentially never
    reaches the target (empirical Pk ~ 0);
  * kills are strictly more likely near than far.

It also writes a CSV (``test_write_range_zone_report``) logging predicted zones
against realized outcomes so the predicted-vs-observed gap — e.g. a predicted
``r_aero`` far larger than the missile's true flyout — is inspectable. The
report feeds the Monte-Carlo calibration in Phase 8; it is not asserted tightly
here because the kill model is not yet calibrated.
"""

import csv
import math
import os
import random
from pathlib import Path

import pytest

from bvr_marl_core.aircraft.core.nez import clear_dlz_cache
from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.missiles.fox3.k77m import K77M
from bvr_marl_core.missiles.fox3.meteor import Meteor
from bvr_marl_core.missiles.fox3.r37m import R37M
from bvr_marl_core.missiles.fox3.r77_1 import R77_1
from bvr_marl_core.missiles.missile_parameters import MissileParameters
from bvr_marl_core.simulator.core.events import MissileTerminalEvent
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator

pytestmark = pytest.mark.slow


MISSILES = {
    "AIM120_AMRAAM": AIM120_AMRAAM,
    "Meteor": Meteor,
    "K77M": K77M,
    "R77_1": R77_1,
    "R37M": R37M,
}

# Target heading (geographic) per launch aspect, with the shooter at the west
# end pointing east (yaw 90).
_ASPECT_YAW = {"head_on": 270.0, "tail": 90.0, "crossing": 0.0}


class _MapLimits:
    bottom_lat = -5.0
    top_lat = 5.0
    left_lon = -5.0
    right_lon = 5.0
    min_alt = 0.0
    max_alt = 20_000.0


def _fighter(lat, lon, yaw, speed, group, alt=8_000.0):
    return Eurofighter(
        Position(lat, lon, alt),
        yaw_deg=yaw,
        speed_mps=speed,
        group=group,
        map_limits=_MapLimits(),
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )


def _set_loadout(shooter, missile_cls):
    """Configure the shooter to carry exactly the missile under test.

    ``missile_params`` is precomputed at aircraft construction and takes
    precedence in ``NoEscapeZoneCalculator._get_best_missile_params``, so it must
    be rebuilt (not just ``missile_types``) for the predicted DLZ to reflect the
    fired missile.
    """
    shooter.missile_types = [missile_cls]
    shooter.missiles = []
    shooter.missile_params = {
        missile_cls.__name__: MissileParameters.from_missile_class(missile_cls)
    }


def _place_target_east(shooter, slant_m, alt_t, yaw_t, spd_t, group="red"):
    """Place a target due east of the shooter at the requested slant range."""
    dz = alt_t - shooter.position.alt
    horiz = math.sqrt(max(slant_m**2 - dz**2, 0.0))
    cos_lat = math.cos(math.radians(shooter.position.lat))
    dlon = horiz / (111_000.0 * cos_lat)
    return _fighter(shooter.position.lat, shooter.position.lon + dlon, yaw_t, spd_t, group, alt_t)


def run_engagement(
    missile_cls,
    slant_m,
    seed,
    *,
    aspect="head_on",
    alt_s=8_000.0,
    alt_t=8_000.0,
    spd_s=300.0,
    spd_t=250.0,
    maneuver=None,
    ticks=160,
):
    """Fire one missile at a controlled geometry and return ``(dlz, record)``.

    ``record`` is the harvested terminal-event dict, or ``None`` if the missile
    never detonated (ran out of energy / lifetime first).
    """
    sim = Simulator(tick_secs=1.0, random_seed=seed)
    shooter = _fighter(0.0, 0.0, 90.0, spd_s, "blue", alt_s)
    _set_loadout(shooter, missile_cls)
    target = _place_target_east(shooter, slant_m, alt_t, _ASPECT_YAW[aspect], spd_t)
    sim.add_unit(shooter)
    sim.add_unit(target)

    # The DLZ cache is a module global keyed by (own_id, target_id); fresh sims
    # reuse those ids, so clear it before reading the predicted zone pre-tick.
    clear_dlz_cache()
    dlz = shooter.wez.compute_dlz(target)

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
    return dlz, record


def _band_outcomes(missile_cls, slant_m, seeds, **kw):
    """Monte-Carlo a launch band: return (shots, detonations, kills, miss_list)."""
    dets = kills = 0
    misses = []
    for seed in range(seeds):
        _, rec = run_engagement(missile_cls, slant_m, seed, **kw)
        if rec is not None:
            dets += 1
            if rec.get("killed"):
                kills += 1
            if rec.get("miss_distance_m") is not None:
                misses.append(rec["miss_distance_m"])
    return seeds, dets, kills, misses


def _zone_for_range(slant_m, dlz):
    """Classify a slant range into the DLZ bands (mirrors
    ``NoEscapeZoneCalculator.zone_for_range``, which is a method on the
    calculator rather than the DLZ dataclass)."""
    if slant_m < dlz.r_min_m:
        return "R1"
    if slant_m < dlz.r_tr_m:
        return "R2"
    if slant_m < dlz.r_pi_m:
        return "R3"
    return "R4"


def _near_range(dlz):
    """A modest range comfortably inside the practical band and within any
    BVR missile's true flyout."""
    return max(dlz.r_min_m + 3_000.0, min(20_000.0, dlz.r_min_m + 0.2 * (dlz.r_pi_m - dlz.r_min_m)))


@pytest.mark.parametrize("name,missile_cls", list(MISSILES.items()))
def test_range_zone_correctness(name, missile_cls):
    """Robust physical correctness (holds pre-calibration): a reachable shot
    can kill, an absurdly-long shot never reaches, and kills are more likely
    near than far."""
    seeds = int(os.environ.get("BVR_RANGE_VALIDATION_SEEDS", "4"))

    dlz, _ = run_engagement(missile_cls, 20_000.0, 0)  # baseline DLZ for this geometry

    near = _near_range(dlz)
    # Beyond *any* BVR missile's true flyout (lifetime/energy bound), independent
    # of the — possibly miscalibrated — predicted r_aero.
    far = max(dlz.r_aero_m * 1.5, 500_000.0)

    _, dets_near, kills_near, _ = _band_outcomes(missile_cls, near, seeds)
    _, _, kills_far, _ = _band_outcomes(missile_cls, far, seeds)

    assert dets_near >= 1, f"{name}: no detonation at a reachable {near / 1000:.0f} km"
    assert kills_near >= 1, f"{name}: never killed at a reachable {near / 1000:.0f} km"
    assert kills_far == 0, f"{name}: scored a kill at an unreachable {far / 1000:.0f} km"
    assert kills_near > kills_far


@pytest.mark.parametrize("name,missile_cls", list(MISSILES.items()))
def test_non_maneuvering_head_on_target_is_intercepted_from_80km(name, missile_cls):
    """An unopposed 80 km head-on shot must reach a fighter-sized target."""
    _, record = run_engagement(
        missile_cls,
        80_000.0,
        seed=0,
        aspect="head_on",
        maneuver=None,
        ticks=180,
    )

    assert record is not None, f"{name}: no terminal event from an 80 km head-on shot"
    assert record["geometric_hit"], f"{name}: closest approach exceeded its lethal radius"
    assert record["miss_distance_m"] < record["lethal_radius_m"]


@pytest.mark.parametrize("name,missile_cls", list(MISSILES.items()))
def test_predicted_aero_bounds_flyout(name, missile_cls):
    """The predicted aerodynamic range bounds where the missile can actually kill.

    After the DLZ recalibration (r_aero anchored on each missile's cited kinematic
    range and tracking its simulated flyout), a shot comfortably beyond the
    predicted r_aero no longer scores kills — the predicted zone is now consistent
    with the flyout model.
    """
    seeds = int(os.environ.get("BVR_RANGE_VALIDATION_SEEDS", "4"))
    dlz, _ = run_engagement(missile_cls, 20_000.0, 0)
    beyond_predicted = dlz.r_aero_m * 1.2
    _, _, kills_beyond, _ = _band_outcomes(missile_cls, beyond_predicted, seeds)
    assert kills_beyond == 0, (
        f"{name}: killed beyond predicted r_aero ({beyond_predicted / 1000:.0f} km) — "
        "predicted aerodynamic range does not bound true flyout"
    )


def _csv_path() -> Path:
    configured = os.environ.get("BVR_RANGE_VALIDATION_CSV")
    if configured:
        return Path(configured)
    return Path("output") / "missile_telemetry" / "range_zone_validation.csv"


def test_write_range_zone_report():
    """Sweep range bands × aspects per missile and log predicted zones against
    realized outcomes for inspection / Phase-8 calibration."""
    seeds = int(os.environ.get("BVR_RANGE_VALIDATION_REPORT_SEEDS", "3"))
    aspects = ("head_on", "tail", "crossing")
    bands = ("inside_min", "near", "mid", "beyond_aero")

    rows = []
    for name, missile_cls in MISSILES.items():
        dlz, _ = run_engagement(missile_cls, 20_000.0, 0)
        band_ranges = {
            "inside_min": 0.6 * dlz.r_min_m,
            "near": max(dlz.r_min_m + 3_000.0, min(20_000.0, 0.2 * dlz.r_pi_m)),
            "mid": 0.5 * (dlz.r_min_m + dlz.r_pi_m),
            "beyond_aero": dlz.r_aero_m * 1.5,
        }
        for band in bands:
            slant = band_ranges[band]
            for aspect in aspects:
                maneuver = "jink" if aspect == "head_on" else None
                shots, dets, kills, misses = _band_outcomes(
                    missile_cls, slant, seeds, aspect=aspect, maneuver=maneuver
                )
                pk_obs = kills / shots if shots else 0.0
                predicted_zone = _zone_for_range(slant, dlz)
                rows.append(
                    {
                        "missile": name,
                        "band": band,
                        "aspect": aspect,
                        "maneuver": maneuver or "none",
                        "R_launch_m": round(slant, 1),
                        "r_min_m": round(dlz.r_min_m, 1),
                        "r_nez_out_m": round(dlz.r_nez_out_m, 1),
                        "r_pi_m": round(dlz.r_pi_m, 1),
                        "r_aero_m": round(dlz.r_aero_m, 1),
                        "predicted_zone": predicted_zone,
                        "shots": shots,
                        "detonations": dets,
                        "kills": kills,
                        "pk_observed": round(pk_obs, 3),
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
    # The report is diagnostic — its purpose is to expose where predicted zones
    # disagree with realized outcomes (e.g. kills beyond a too-short predicted
    # r_aero), not to enforce calibration. Only sanity-check that the harness
    # produced kills somewhere.
    assert any(r["kills"] > 0 for r in rows)
