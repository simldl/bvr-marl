"""
Missile guidance accuracy test for maneuvering targets.

Tests the PN / ZEM guidance chain against various target maneuver profiles and
reports the closest approach (miss distance) as well as hit outcome.

Key findings from branch comparison (main vs scripted-behavior):
  - main branch bugs (fixed in scripted-behavior):
      1. Wrong ENU rotation direction in pn_propnav._enu_rotation
         (used B_to @ B_from.T instead of B_to.T @ B_from).
         This caused the velocity to be rotated in the WRONG direction,
         so a beam-maneuvering target's East velocity appeared as North, etc.
      2. QR decomposition on _enu_basis introduced arbitrary sign flips that
         further corrupted the rotation matrix.
      3. Finite-difference fallback: old code subtracted _prev_r (in ENU at
         missile position at t-1) from current R_vec (ENU at t). Since the
         missile moved, the two vectors are in different ENU frames, making
         the difference physically meaningless.
      4. Same QR sign-flip bug in radar/tracking/helpers/enu_utils.py was
         also fixed, so tracker ENU transforms are now consistent.

  - Remaining weaknesses (still in scripted-behavior branch):
      A. Kalman filter velocity converges slowly: for the first 2-3 s after
         launch (or after the target starts maneuvering), the KF velocity
         estimate is near zero.  Lead guidance falls back to direct pursuit.
      B. Phase selection: PN only activates when the target is inside the
         missile radar FOV (60° cone) AND the missile has a track lock. A fast
         lateral mover can drift outside FOV, forcing direct fallback
         (position-only, no velocity) or lead with possibly stale velocity.
      C. ZEM t_go estimate assumes constant closing speed (R / Vc). This is
         accurate only when missile speed is stable (after boost). During the
         boost phase the true t_go is shorter, so ZEM overshoots slightly.
      D. "Hold last valid velocity" (3 s max) keeps guidance working during
         brief tracker outages, but if the target makes a hard turn within
         that 3 s window the held velocity is completely stale.
      E. At long range (>30 km with F22 radar settings) the shooter's radar
         cannot detect the target, so no datalink data reaches the missile.
         The missile is then stuck without a usable guidance position.

Test methodology:
  - Direct-creation of AIM120_AMRAAM within a Simulator (both shooter and
    target are F22, range is kept ≤20 km where radar reliably locks).
  - Shooter warms up for radar lock acquisition before missile is fired.
  - Target maneuver is applied each tick via control.set_yaw_deg().
  - Closest approach distance is tracked every tick (geodetic, 3-D).
  - Hit is declared when the target leaves sim.active_units.
  - A miss is declared when the missile leaves sim.active_units.
  - Tests pass when hit rates meet expected thresholds OR when miss distance
    is below a diagnostic threshold (to catch regressions without hardcoding
    perfect accuracy).
"""

import math
import random

import numpy as np
import pytest

from bvr_marl_core.aircraft.types.f22 import F22
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MapLimits:
    bottom_lat = -5.0
    top_lat = 5.0
    left_lon = -5.0
    right_lon = 5.0
    min_alt = 0
    max_alt = 20_000


def _dist3d(p1, p2) -> float:
    """Approximate 3-D distance between two geodetic Positions (metres)."""
    dlat = (p2.lat - p1.lat) * 111_000.0
    dlon = (p2.lon - p1.lon) * 111_000.0 * math.cos(math.radians(0.5 * (p1.lat + p2.lat)))
    dalt = p2.alt - p1.alt
    return math.sqrt(dlat * dlat + dlon * dlon + dalt * dalt)


def _ccd_min_dist(m0, t0, m1, t1) -> float:
    """
    Continuous Collision Detection — minimum 3-D distance between missile and
    target during a single tick, using linear interpolation of positions.

    This matches what MissileHitCalculator does internally, so the reported
    closest_m is always the true minimum approach distance, not just the
    end-of-tick snapshot which can be 100-150 m higher than the real minimum.
    """
    lat_ref = 0.5 * (m0.lat + t0.lat)
    lon_ref = 0.5 * (m0.lon + t0.lon)
    lat_scale = 111_000.0
    lon_scale = 111_000.0 * math.cos(math.radians(lat_ref))

    def to_m(p):
        return np.array(
            [(p.lon - lon_ref) * lon_scale, (p.lat - lat_ref) * lat_scale, p.alt], dtype=float
        )

    pm0, pm1 = to_m(m0), to_m(m1)
    pt0, pt1 = to_m(t0), to_m(t1)
    r0 = pm0 - pt0
    v_rel = (pm1 - pm0) - (pt1 - pt0)
    denom = float(v_rel @ v_rel)
    if denom < 1e-10:
        return float(np.linalg.norm(r0))
    t_star = max(0.0, min(1.0, float(-(r0 @ v_rel) / denom)))
    return float(np.linalg.norm(r0 + t_star * v_rel))


def run_engagement(
    range_km: float,
    target_speed_mps: float,
    maneuver: str,
    seed: int = 0,
    jink_interval_s: float = 3.0,
    warmup_ticks: int = 60,
    max_ticks: int = 1_500,
    tick_secs: float = 0.1,
) -> dict:
    """
    Run a single head-on engagement and return result dict with keys:
        hit         : bool
        closest_m   : float  — minimum 3-D distance between missile and target
        time_s      : float  — elapsed simulation time at engagement end
        phase_dist  : dict   — {phase_name: tick_count}

    Parameters
    ----------
    range_km        Initial slant range (head-on, same altitude).
    target_speed_mps  Target TAS.
    maneuver        One of: 'head_on', 'beam', 'cold', 'jink'
    seed            RNG seed for jinking.
    jink_interval_s Seconds between random heading changes for 'jink'.
    warmup_ticks    Ticks before firing (for radar lock acquisition).
    """
    rng = random.Random(seed)
    ml = _MapLimits()

    range_deg = range_km / 111.0

    shooter = F22(
        Position(0.0, 0.0, 8_000),
        yaw_deg=90.0,
        speed_mps=300.0,
        group="blue",
        map_limits=ml,
        min_alt_m=0,
        max_alt_m=20_000,
    )
    target = F22(
        Position(0.0, range_deg, 8_000),
        yaw_deg=270.0,
        speed_mps=target_speed_mps,
        group="red",
        map_limits=ml,
        min_alt_m=0,
        max_alt_m=20_000,
    )

    sim = Simulator(tick_secs=tick_secs)
    sim.add_unit(shooter)
    sim.add_unit(target)

    # Warm up – wait for shooter radar to acquire a lock so the missile gets
    # initial track data from the datalink when it is created.
    lock_achieved = False
    for _ in range(warmup_ticks):
        sim.do_tick()
        if shooter.sensor.get_locked_targets():
            lock_achieved = True
            break

    missile = AIM120_AMRAAM(
        firing_time_s=sim.elapsed_time_s,
        target=target,
        source=shooter,
        map_limits=ml,
        group="blue",
    )
    sim.add_unit(missile)

    # Patch guidance to record which phase is chosen each tick
    guidance = missile.guidance
    phase_dist: dict[str, int] = {}
    _orig_sel = guidance._select_guidance_phase

    def _counting_select(m, tp, tm, dt):
        ph = _orig_sel(m, tp, tm, dt)
        phase_dist[ph] = phase_dist.get(ph, 0) + 1
        return ph

    guidance._select_guidance_phase = _counting_select

    closest_m = 1e9
    end_time_s = max_ticks * tick_secs
    last_jink = 0.0
    hit = False

    for tick in range(max_ticks):
        t = tick * tick_secs

        # Apply target maneuver
        if maneuver == "head_on":
            target.control.set_yaw_deg(270.0)  # fly toward shooter
        elif maneuver == "beam":
            target.control.set_yaw_deg(0.0)  # fly perpendicular (North)
        elif maneuver == "cold":
            target.control.set_yaw_deg(90.0)  # fly away from shooter
        elif maneuver == "jink":
            if t - last_jink >= jink_interval_s:
                target.control.set_yaw_deg(rng.uniform(0.0, 360.0))
                last_jink = t

        # Snapshot pre-tick positions for CCD-accurate closest approach
        m_pos_pre = missile.position if missile.id in sim.active_units else None
        t_pos_pre = target.position if target.id in sim.active_units else None

        sim.do_tick()

        missile_alive = missile.id in sim.active_units
        target_alive = target.id in sim.active_units

        # CCD-accurate closest approach during this tick — including the detonation tick
        # (missile.position is still valid even after removal from active_units)
        if m_pos_pre is not None and t_pos_pre is not None:
            d = _ccd_min_dist(m_pos_pre, t_pos_pre, missile.position, target.position)
            if d < closest_m:
                closest_m = d

        if not missile_alive:
            # Lethal hits enter a short death-spiral state before the aircraft is
            # removed, so immediate roster membership alone undercounts kills.
            hit = bool(getattr(target, "is_mortally_hit", False)) or not target_alive
            end_time_s = t
            break

        if not target_alive:
            hit = True
            end_time_s = t
            break

    return {
        "hit": hit,
        "closest_m": closest_m,
        "time_s": end_time_s,
        "phase_dist": phase_dist,
        "lock_at_launch": lock_achieved,
    }


def _multi_run(n: int, **kwargs) -> dict:
    """Run n engagements with seeds 0..n-1 and aggregate results."""
    results = [run_engagement(seed=i, **kwargs) for i in range(n)]
    hits = sum(r["hit"] for r in results)
    closests = [r["closest_m"] for r in results]
    within_radius = sum(r["closest_m"] <= FUSE_RADIUS_M for r in results)
    return {
        "hits": hits,
        "within_radius": within_radius,
        "n": n,
        "hit_rate": hits / n,
        "closest_mean_m": float(np.mean(closests)),
        "closest_max_m": float(np.max(closests)),
        "closest_min_m": float(np.min(closests)),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

RANGE_KM = 18.0  # within F22 radar detection range
N_RUNS = 5  # seeds per scenario
FUSE_RADIUS_M = 500.0  # AIM-120 fuse radius (matches HitParams default)


class TestGuidanceBaseline:
    """Head-on non-maneuvering target — sanity check that guidance works at all."""

    def test_head_on_300(self):
        """300 m/s head-on should bring missile within fuse radius reliably."""
        out = _multi_run(N_RUNS, range_km=RANGE_KM, target_speed_mps=300.0, maneuver="head_on")
        # At least 3/5 runs must reach within fuse radius (deterministic geometry check)
        assert out["within_radius"] >= 3, (
            f"Head-on 300 m/s: only {out['within_radius']}/{N_RUNS} runs within {FUSE_RADIUS_M:.0f} m. "
            f"Mean closest: {out['closest_mean_m']:.0f} m. "
            "PN guidance may be broken."
        )

    def test_head_on_500(self):
        """Higher-speed head-on — closing speed rises, missile should still reach fuse radius."""
        out = _multi_run(N_RUNS, range_km=RANGE_KM, target_speed_mps=500.0, maneuver="head_on")
        assert out["within_radius"] >= 3, (
            f"Head-on 500 m/s: only {out['within_radius']}/{N_RUNS} runs within {FUSE_RADIUS_M:.0f} m. "
            f"Mean closest: {out['closest_mean_m']:.0f} m."
        )


class TestGuidanceBeamTarget:
    """Beam (90° crossing) target — exercises lateral velocity prediction."""

    def test_beam_300_miss_distance(self):
        """
        300 m/s beam target: guidance must predict intercept correctly.
        If the missile aims at 'current position' instead of predicted intercept
        the miss distance is proportional to target speed × flight time (can be km).
        We require closest approach < 2 km to catch the 'past position' regression.
        """
        out = _multi_run(N_RUNS, range_km=RANGE_KM, target_speed_mps=300.0, maneuver="beam")
        assert out["closest_mean_m"] < 2_000, (
            f"Beam 300 m/s: mean miss {out['closest_mean_m']:.0f} m > 2000 m. "
            "Guidance is likely ignoring target velocity (aiming at past position). "
            f"Phase dist: {out['results'][0]['phase_dist']}"
        )

    def test_beam_500_miss_distance(self):
        """500 m/s beam target: intercept computation must handle fast movers."""
        out = _multi_run(N_RUNS, range_km=RANGE_KM, target_speed_mps=500.0, maneuver="beam")
        assert out["closest_mean_m"] < 3_000, (
            f"Beam 500 m/s: mean miss {out['closest_mean_m']:.0f} m > 3000 m. "
            "Fast lateral mover exposes velocity-prediction weakness."
        )

    def test_beam_300_hit_rate(self):
        """Beam target at moderate speed: missile should reach fuse radius in most runs."""
        out = _multi_run(N_RUNS, range_km=RANGE_KM, target_speed_mps=300.0, maneuver="beam")
        assert out["within_radius"] >= 2, (
            f"Beam 300 m/s: only {out['within_radius']}/{N_RUNS} runs within {FUSE_RADIUS_M:.0f} m. "
            f"Closest approaches: {[round(r['closest_m']) for r in out['results']]}"
        )


class TestGuidanceJinkingTarget:
    """Randomly jinking target — exercises velocity hold and phase robustness."""

    def test_jink_slow_interval(self):
        """Jinking every 5 s — filter should converge between turns."""
        out = _multi_run(
            N_RUNS,
            range_km=RANGE_KM,
            target_speed_mps=400.0,
            maneuver="jink",
            jink_interval_s=5.0,
        )
        # Jinking should not completely defeat guidance
        assert out["closest_mean_m"] < 5_000, (
            f"Jink-5s: mean miss {out['closest_mean_m']:.0f} m > 5000 m."
        )

    def test_jink_fast_interval(self):
        """Jinking every 2 s — harder; at least some hits or close misses."""
        out = _multi_run(
            N_RUNS,
            range_km=RANGE_KM,
            target_speed_mps=400.0,
            maneuver="jink",
            jink_interval_s=2.0,
        )
        # We expect at least 1 run within fuse radius or all miss distances < 3 km
        some_within_radius = out["within_radius"] >= 1
        all_close = out["closest_max_m"] < 3_000
        assert some_within_radius or all_close, (
            f"Jink-2s: {out['within_radius']}/{N_RUNS} runs within {FUSE_RADIUS_M:.0f} m, "
            f"max miss {out['closest_max_m']:.0f} m. "
            "Guidance completely lost the target during jinking."
        )


class TestGuidancePhaseDistribution:
    """Verify PN phase is actually used (not just direct/lead throughout)."""

    def test_pn_used_for_head_on(self):
        """PN should dominate the engagement for a head-on non-maneuvering target."""
        result = run_engagement(
            range_km=RANGE_KM,
            target_speed_mps=300.0,
            maneuver="head_on",
            seed=0,
        )
        pd = result["phase_dist"]
        total = sum(pd.values()) or 1
        pn_frac = pd.get("pn", 0) / total
        assert pn_frac > 0.3, (
            f"PN only used {pn_frac:.1%} of ticks (expected >30%). "
            f"Full phase dist: {pd}. "
            "Guidance may be stuck in direct fallback (no tracker data)."
        )

    def test_pn_or_lead_used_for_beam(self):
        """For a beam target, at least some PN or lead guidance should be used."""
        result = run_engagement(
            range_km=RANGE_KM,
            target_speed_mps=300.0,
            maneuver="beam",
            seed=0,
        )
        pd = result["phase_dist"]
        total = sum(pd.values()) or 1
        pn_lead_frac = (pd.get("pn", 0) + pd.get("lead", 0)) / total
        assert pn_lead_frac > 0.2, (
            f"PN+lead only {pn_lead_frac:.1%} of ticks. "
            f"Full dist: {pd}. "
            "Missile stuck in direct fallback - velocity not being used."
        )


class TestVelocityEstimation:
    """Directly probe the PN velocity estimate to check it's non-zero and sane."""

    def test_velocity_converges_before_terminal_phase(self):
        """
        The tracker KF must converge to a non-zero velocity estimate before the
        missile enters the terminal phase (< 2 km slant range).  A zero velocity
        at terminal phase causes pure-pursuit and lateral misses.
        """
        ml = _MapLimits()
        range_deg = RANGE_KM / 111.0
        shooter = F22(Position(0.0, 0.0, 8_000), 90.0, 300.0, "blue", ml, 0, 20_000)
        target = F22(Position(0.0, range_deg, 8_000), 0.0, 400.0, "red", ml, 0, 20_000)

        sim = Simulator(tick_secs=0.1)
        sim.add_unit(shooter)
        sim.add_unit(target)
        for _ in range(60):
            sim.do_tick()

        missile = AIM120_AMRAAM(0.0, target, shooter, ml, "blue")
        sim.add_unit(missile)

        pn = missile.guidance.pn
        velocity_at_terminal: list[float] = []

        for tick in range(1_500):
            target.control.set_yaw_deg(0.0)  # beam
            sim.do_tick()

            if missile.id not in sim.active_units or target.id not in sim.active_units:
                break

            # Check if we are in terminal phase (< 2 km slant to target)
            dist_m = _dist3d(missile.position, target.position)
            if dist_m < 2_000:
                v = pn._last_valid_vt
                if v is not None:
                    velocity_at_terminal.append(float(np.linalg.norm(v)))

        # If we have any terminal-phase velocity readings, they should be > 50 m/s
        # (a beam target doing 400 m/s would have ENU velocity magnitude of ~400 m/s)
        if velocity_at_terminal:
            mean_vel = float(np.mean(velocity_at_terminal))
            assert mean_vel > 50.0, (
                f"Mean velocity estimate in terminal phase: {mean_vel:.1f} m/s < 50 m/s. "
                "KF has not converged — guidance will aim at current position, not intercept. "
                "This reproduces the 'flying toward past position' symptom."
            )

    def test_velocity_nonzero_at_mid_course(self):
        """
        At least 3 s into the engagement, the target provider's velocity estimate
        should be non-zero for a beam-maneuvering target.
        """
        ml = _MapLimits()
        range_deg = RANGE_KM / 111.0
        shooter = F22(Position(0.0, 0.0, 8_000), 90.0, 300.0, "blue", ml, 0, 20_000)
        target = F22(Position(0.0, range_deg, 8_000), 0.0, 400.0, "red", ml, 0, 20_000)

        sim = Simulator(tick_secs=0.1)
        sim.add_unit(shooter)
        sim.add_unit(target)
        for _ in range(60):
            sim.do_tick()

        missile = AIM120_AMRAAM(0.0, target, shooter, ml, "blue")
        sim.add_unit(missile)

        for tick in range(1_500):
            target.control.set_yaw_deg(0.0)
            sim.do_tick()
            if missile.id not in sim.active_units or target.id not in sim.active_units:
                break
            # Check velocity after 3 s of flight
            if missile.elapsed_time_s > 3.0:
                vel = missile.target_provider.get_guidance_velocity()
                if vel is not None:
                    vmag = float(np.linalg.norm(vel))
                    assert vmag > 30.0, (
                        f"After 3 s, velocity estimate = {vmag:.1f} m/s < 30 m/s. "
                        "KF has not converged. Target is 400 m/s beam mover."
                    )
                break


# ---------------------------------------------------------------------------
# Diagnostic runner (not a pytest test, run with python ... for rich output)
# ---------------------------------------------------------------------------


def _print_summary():
    """
    Print a rich human-readable summary of guidance performance across a range
    of maneuver types, speeds, and ranges.

    Run with:  python tests/missiles/test_guidance_vs_maneuvering_target.py
    """
    import textwrap

    header = "\n=" * 70 + "\n MISSILE GUIDANCE ACCURACY -- MANEUVERING TARGET DIAGNOSTIC\n=" * 70
    print(header)
    print(
        textwrap.dedent("""\
        Setup: AIM-120 AMRAAM vs F-22  |  initial range 18 km  |  alt 8 km
               Shooter F22 warms up 6 s before launch.
               Fuse radius = 500 m  (eff_radius from hit_calculation.py)
               5 seeds per scenario.
        """)
    )

    configs = [
        # (speed, maneuver, jink_s, label)
        (300, "head_on", 3.0, "300 m/s  head-on (baseline)"),
        (300, "beam", 3.0, "300 m/s  beam   (lateral)"),
        (400, "beam", 3.0, "400 m/s  beam   (lateral)"),
        (500, "beam", 3.0, "500 m/s  beam   (fast lat)"),
        (300, "cold", 3.0, "300 m/s  cold   (tail-chase)"),
        (300, "jink", 5.0, "300 m/s  jink-5s"),
        (400, "jink", 3.0, "400 m/s  jink-3s"),
        (400, "jink", 2.0, "400 m/s  jink-2s (hard)"),
        (500, "jink", 2.0, "500 m/s  jink-2s (hardest)"),
    ]

    fmt_hdr = f"{'Scenario':32} {'H/N':>5}  {'Mean-miss':>9}  {'Max-miss':>9}  {'Phases (top-3)'}"
    print(fmt_hdr)
    print("-" * 80)

    for spd, man, jint, lbl in configs:
        out = _multi_run(
            N_RUNS,
            range_km=RANGE_KM,
            target_speed_mps=spd,
            maneuver=man,
            jink_interval_s=jint,
        )
        # Aggregate phase dist
        agg_phases: dict[str, int] = {}
        for r in out["results"]:
            for ph, cnt in r["phase_dist"].items():
                agg_phases[ph] = agg_phases.get(ph, 0) + cnt
        total_ticks = sum(agg_phases.values()) or 1
        top3 = sorted(agg_phases.items(), key=lambda x: -x[1])[:3]
        phase_str = ", ".join(f"{ph}:{cnt / total_ticks:.0%}" for ph, cnt in top3)

        hits = out["hits"]
        total = out["n"]
        mean_m = out["closest_mean_m"]
        max_m = out["closest_max_m"]
        print(f"{lbl:32} {hits}/{total}  {mean_m:9.0f}m  {max_m:9.0f}m  {phase_str}")

    print("\n" + "=" * 70)
    print(
        textwrap.dedent("""\
        INTERPRETATION:
          Mean-miss > 2000 m for beam/jink -> guidance is NOT predicting target
          position correctly (velocity estimation failure or wrong ENU rotation).
          If 'pn' phase fraction is < 20 %  -> tracker is not providing lock /
          velocity, missile is stuck in direct fallback (position-only).
        """)
    )


if __name__ == "__main__":
    _print_summary()
