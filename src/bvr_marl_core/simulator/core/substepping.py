from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil

from bvr_marl_core.simulator.core.events import (
    MissileEngagementEvent,
    MissileEnteredTerminalRegionEvent,
)
from bvr_marl_core.simulator.core.helpers import clamp, units_distance_km
from bvr_marl_core.simulator.utils.tick_path import pose_at_fraction, unit_tick_path


@dataclass
class SubstepConfig:
    """Adaptive substepping near target."""

    engage_distance_km: float = 2.0
    min_substeps: int = 4
    max_substeps: int = 16
    safety_travel_per_substep_m: float = 250.0
    min_dt: float = 1e-3
    physics_only: bool = False


class TerminalPathResolver:
    """Authoritative proximity-fuze resolver over the missile's recorded path.

    Missile guidance records the fine path it actually integrated. This resolver
    walks that path once against a cubic-Hermite target path reconstructed from
    begin/end position and velocity. It performs no rewind or replay and delegates
    the sole detonation/lethality decision to ``MissileCCDManager.on_hit``.
    """

    def __init__(self, config: SubstepConfig | None = None) -> None:
        self.cfg = config or SubstepConfig()

    def run_tick_with_substeps(self, sim, tick_secs: float):
        self._candidate_ranges_m = {}
        pairs = self._collect_candidates(sim.active_units.values(), sim)
        if not pairs:
            return []

        ccd = getattr(sim, "ccd", None)
        calc = getattr(ccd, "hit_calc", None) if ccd is not None else None
        if calc is None:
            calc = getattr(sim, "hit_calc", None)
        if calc is None:
            return []

        for missile, target in pairs:
            self._substep_pair(sim, missile, target, tick_secs, calc)

        return []

    def _collect_candidates(
        self, units: Iterable[object], sim: object | None = None
    ) -> list[tuple[object, object]]:
        out: list[tuple[object, object]] = []
        for u in units:
            if not self._is_missile(u):
                continue
            if (
                getattr(u, "should_be_removed", False)
                and getattr(u, "removal_reason", None) == "target_destroyed"
            ):
                continue
            # Resolve the target the missile is actually guiding on. Real missiles
            # carry no `locked_target` object attribute — the lock lives on the
            # radar as a unit ID — so resolve that ID to a unit via sim as the
            # authoritative source, falling back to the missile's target object.
            operational_track = getattr(u, "weapon_track", None) is not None
            evaluator_resolver = getattr(sim, "evaluator_target_for_weapon", None)
            evaluator_target = (
                evaluator_resolver(u)
                if isinstance(getattr(sim, "_weapon_truth_associations", None), dict)
                and callable(evaluator_resolver)
                else None
            )
            if operational_track:
                # Track IDs never enter the physical-unit namespace. Missing
                # evaluator attribution therefore fails closed.
                tgt = evaluator_target
                if tgt is None:
                    record = getattr(sim, "record_diagnostic", None)
                    departed = getattr(sim, "evaluator_weapon_target_departed", None)
                    if callable(departed) and departed(u):
                        u.should_be_removed = True
                        u.removal_reason = "target_destroyed"
                        if callable(record):
                            record("weapon_target_departed")
                    elif callable(record):
                        record("missing_evaluator_target")
            else:
                # Explicit legacy/oracle missiles retain object/Unit-ID lookup.
                tgt = (
                    evaluator_target
                    or getattr(u, "locked_target", None)
                    or self._resolve_radar_lock_target(u, sim)
                    or getattr(u, "target", None)
                )
            if tgt is None or getattr(tgt, "is_destroyed", False):
                continue
            try:
                distance_km = units_distance_km(u, tgt)
                # Once a missile has entered the terminal region, keep resolving
                # the pair while it opens again.  Restricting candidates to the
                # current range drops the pair immediately after an outside-fuze
                # CPA; guidance then turns the still-live missile back toward the
                # aircraft, producing the visible "orbiting" behaviour.
                ccd = getattr(sim, "ccd", None)
                engaged = getattr(ccd, "_engaged_once", None)
                already_terminal = isinstance(engaged, set) and getattr(u, "id", None) in engaged
                if distance_km <= self.cfg.engage_distance_km or already_terminal:
                    out.append((u, tgt))
                    self._candidate_ranges_m[(getattr(u, "id", None), getattr(tgt, "id", None))] = (
                        distance_km * 1000.0
                    )
            except Exception:  # noqa: BLE001 - substep hot loop: a malformed unit must
                # never abort candidate collection for the whole tick. Contract is
                # pinned by test_distance_calculation_exception.
                continue
        return out

    def _substep_pair(self, sim, missile, target, tick_secs: float, calc) -> None:
        """Run continuous-collision detection along the missile's *authoritative*
        sub-stepped path for this tick.

        The missile records the fine positions it actually flew (``tick_path``)
        while integrating its own guidance + movement. We walk those segments
        against the target's own recorded arc for this tick, so the miss distance
        reflects two real trajectories rather than one real and one guessed. The
        first geometric contact is handed to ``ccd.on_hit`` with that miss
        distance (range-dependent Pk).

        When the target records no usable path (a stub, or a platform that does
        not sub-step), we fall back to a velocity-matched Hermite curve through
        its begin- and end-of-tick poses. That fallback is why this used to be
        wrong for the case that matters: it assumes a gently-turning target, and
        a 5 g defensive break at 70 deg of bank departs from it by more than the
        proximity fuze's lethal radius.
        """
        ccd = getattr(sim, "ccd", None)

        missile_id = getattr(missile, "id", None)
        engaged = getattr(ccd, "_engaged_once", None) if ccd is not None else None
        logger = getattr(sim, "log_event", None)
        if isinstance(engaged, set) and callable(logger) and missile_id not in engaged:
            engaged.add(missile_id)
            range_m = getattr(self, "_candidate_ranges_m", {}).get(
                (missile_id, getattr(target, "id", None)), 0.0
            )
            logger(MissileEnteredTerminalRegionEvent(sim, missile, target, range_m))
            logger(MissileEngagementEvent(sim, missile, target))

        # Arming gate: an unarmed fuze cannot detonate yet — the missile flies on.
        arm_time = getattr(missile, "arming_time_s", 0.0)
        if float(getattr(missile, "elapsed_time_s", 0.0)) < float(arm_time):
            return

        path = getattr(missile, "tick_path", None)
        if not path or len(path) < 2:
            # No recorded sub-path (e.g. a non-Missile stub): single-segment CCD.
            self._single_segment_fallback(sim, missile, target, tick_secs, calc)
            return

        target_path = unit_tick_path(target, sim)
        if target_path is None:
            begin_state = ccd.get_begin_state(target) if ccd is not None else None
            t_begin = self._begin_pose(ccd, target) or self._pose(target)
            t_end = self._pose(target)
            v_begin = self._pose_rate(target, begin_state)
            v_end = self._pose_rate(target, None)

        def target_pose(f: float):
            if target_path is not None:
                return pose_at_fraction(target_path, f)
            return self._hermite_pose(t_begin, t_end, v_begin, v_end, tick_secs, f)

        n = len(path) - 1
        eff_radius = float(calc.params.fuse_radius_m + calc.params.target_radius_m)

        # Find the closest point of approach (CPA) over the tick's path. A proximity
        # fuze detonates at the minimum miss distance, not when the missile first
        # crosses the (large) fuse radius — otherwise every shot reports a ~fuse-
        # sized miss and the range-dependent Pk reads ~0.
        best_d = float("inf")
        best_i = -1
        best_ts = 1.0
        for i in range(n):
            f0 = i / n
            f1 = (i + 1) / n
            m0, m1 = path[i], path[i + 1]
            t0 = target_pose(f0)
            t1 = target_pose(f1)
            _is_hit, t_star, d_min = calc.check_segment_hit(m0, m1, t0, t1)
            if d_min < best_d:
                best_d = d_min
                best_i = i
                best_ts = t_star if t_star is not None else 1.0

        # Only detonate once the CPA has actually been reached this tick (the
        # minimum is interior to the path, not still-closing at its final point);
        # otherwise the missile is still approaching and we wait for next tick.
        passed_cpa = best_i >= 0 and (best_i < n - 1 or best_ts < 1.0 - 1e-6)
        if best_d <= eff_radius and passed_cpa:
            tf = (best_i + best_ts) / n
            f0 = best_i / n
            f1 = (best_i + 1) / n
            m_cpa = self._lerp_pose(path[best_i], path[best_i + 1], best_ts)
            t_cpa = target_pose(tf)
            try:
                now_s = float(getattr(sim, "elapsed_time_s", 0.0))
            except (TypeError, ValueError):
                now_s = 0.0
            missile._last_cpa_event = {
                "time_s": now_s + tf * float(tick_secs),
                "t_frac": tf,
                "miss_m": best_d,
                "missile_pose": m_cpa,
                "target_pose": t_cpa,
                "target_id": getattr(target, "id", None),
            }
            if ccd is not None and hasattr(ccd, "on_hit"):
                ccd.on_hit(missile, target, tf, sim, miss_distance_m=best_d)
        elif passed_cpa and best_d > eff_radius:
            # A guided air-to-air missile does not make a 180-degree turn and
            # attack again after passing its target.  Resolve an outside-fuze
            # closest approach as a terminal miss, so it cannot circle the
            # aircraft indefinitely.  The path resolver has already checked the
            # whole authoritative sub-stepped path for this tick, so no hit is
            # discarded here.
            tf = (best_i + best_ts) / n
            try:
                now_s = float(getattr(sim, "elapsed_time_s", 0.0))
            except (TypeError, ValueError):
                now_s = 0.0
            missile._last_cpa_event = {
                "time_s": now_s + tf * float(tick_secs),
                "t_frac": tf,
                "miss_m": best_d,
                "missile_pose": self._lerp_pose(path[best_i], path[best_i + 1], best_ts),
                "target_pose": target_pose(tf),
                "target_id": getattr(target, "id", None),
            }
            missile.should_be_removed = True
            missile.removal_reason = "terminal_miss"
            record = getattr(sim, "record_diagnostic", None)
            if callable(record):
                record("missile_terminal_miss")

    def _single_segment_fallback(self, sim, missile, target, tick_secs: float, calc) -> None:
        """Coarse single-segment CCD over the tick when no sub-path is available."""
        hit, victim, t_frac = calc.check_and_maybe_hit(
            missile=missile, potential_targets=[target], tick_secs=tick_secs
        )
        if hit and victim is not None:
            ccd = getattr(sim, "ccd", None)
            if ccd is not None and hasattr(ccd, "on_hit"):
                ccd.on_hit(missile, victim, t_frac or 0.0, sim)

    @staticmethod
    def _begin_pose(ccd, unit) -> tuple[float, float, float] | None:
        """Begin-of-tick (lat, lon, alt) for a unit from the CCD snapshot, or None."""
        if ccd is None or not hasattr(ccd, "get_begin_state"):
            return None
        st = ccd.get_begin_state(unit)
        if not st:
            return None
        p = st.get("pos")
        if p is None:
            return None
        return (float(p.lat), float(p.lon), float(p.alt))

    @staticmethod
    def _lerp_pose(a, b, f: float) -> tuple[float, float, float]:
        return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f)

    @staticmethod
    def _pose_rate(unit, state) -> tuple[float, float, float]:
        if state is not None:
            speed = float(state.get("speed", 0.0))
            yaw = float(state.get("yaw", 0.0))
            pitch = float(state.get("pitch", 0.0))
        else:
            speed = float(getattr(unit, "speed", 0.0))
            yaw = float(getattr(unit, "yaw_deg", 0.0))
            pitch = float(getattr(unit, "pitch_deg", 0.0))
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        horizontal = speed * math.cos(pitch_rad)
        east = horizontal * math.sin(yaw_rad)
        north = horizontal * math.cos(yaw_rad)
        up = speed * math.sin(pitch_rad)
        latitude = float(getattr(getattr(unit, "position", None), "lat", 0.0))
        return (
            north / 111_320.0,
            east / max(111_320.0 * math.cos(math.radians(latitude)), 1.0),
            up,
        )

    @staticmethod
    def _hermite_pose(p0, p1, rate0, rate1, duration_s: float, f: float):
        f = max(0.0, min(1.0, float(f)))
        h00 = 2.0 * f**3 - 3.0 * f**2 + 1.0
        h10 = f**3 - 2.0 * f**2 + f
        h01 = -2.0 * f**3 + 3.0 * f**2
        h11 = f**3 - f**2
        return tuple(
            h00 * p0[i] + h10 * duration_s * rate0[i] + h01 * p1[i] + h11 * duration_s * rate1[i]
            for i in range(3)
        )

    @staticmethod
    def _pose(u) -> tuple[float, float, float]:
        p = u.position
        return (float(p.lat), float(p.lon), float(p.alt))

    @staticmethod
    def _resolve_radar_lock_target(missile: object, sim: object | None) -> object | None:
        """Resolve a missile's live radar lock (a unit ID) to the unit object.

        Returns None when there is no sim, no radar lock, or the locked unit is
        no longer active. Lock IDs share the unit-ID namespace (active_units keys),
        so the lookup is a direct dict get.
        """
        if sim is None:
            return None
        radar = getattr(missile, "radar", None)
        if radar is None or not hasattr(radar, "get_locked_target"):
            return None
        try:
            locked_id = radar.get_locked_target()
        except Exception:
            return None
        if locked_id is None:
            return None
        return getattr(sim, "active_units", {}).get(locked_id)

    @staticmethod
    def _is_missile(u: object) -> bool:
        if getattr(u, "is_destroyed", False):
            return False
        if getattr(u, "is_missile", False):
            return True
        if getattr(u, "unit_type", "").lower() == "missile":
            return True
        return type(u).__name__.lower().endswith("missile")

    def _estimate_substeps(self, missile, target, tick_secs: float) -> int:
        """Estimate number of substeps based on relative speed and config."""
        rel_speed = self._estimate_relative_speed_mps(missile, target)
        if rel_speed < 1e-6:
            return self.cfg.min_substeps
        distance_per_tick = rel_speed * tick_secs
        estimated = ceil(distance_per_tick / self.cfg.safety_travel_per_substep_m)
        return clamp(estimated, self.cfg.min_substeps, self.cfg.max_substeps)

    def _estimate_relative_speed_mps(self, a, b) -> float:
        for attr in ("vel_enu", "velocity"):
            va = getattr(a, attr, None)
            vb = getattr(b, attr, None)
            if va is not None and vb is not None:
                try:
                    ax = float(getattr(va, "vx", va[0]))
                    ay = float(getattr(va, "vy", va[1]))
                    az = float(getattr(va, "vz", va[2]))
                    bx = float(getattr(vb, "vx", vb[0]))
                    by = float(getattr(vb, "vy", vb[1]))
                    bz = float(getattr(vb, "vz", vb[2]))
                    dv0, dv1, dv2 = (ax - bx), (ay - by), (az - bz)
                    return (dv0 * dv0 + dv1 * dv1 + dv2 * dv2) ** 0.5
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    IndexError,
                    ZeroDivisionError,
                ):
                    pass
        sa = float(getattr(a, "speed", 0.0) or 0.0)
        sb = float(getattr(b, "speed", 0.0) or 0.0)
        return abs(sa - sb)

    def _physics_only_step(self, sim, unit, dt: float) -> None:
        fn = getattr(unit, "physics_step", None) or getattr(unit, "update_physics", None)
        if fn is None:
            raise RuntimeError(
                f"{unit} has no physics_step/update_physics - please implement physics-only method."
            )
        fn(dt, sim)
