from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from typing import Optional

from bvr_marl_core.simulator.core.helpers import clamp, units_distance_km


@dataclass
class SubstepConfig:
    """Adaptive substepping near target."""

    engage_distance_km: float = 2.0
    min_substeps: int = 4
    max_substeps: int = 16
    safety_travel_per_substep_m: float = 250.0
    min_dt: float = 1e-3
    physics_only: bool = False


class Substepper:
    """Resolves missile <-> target hits at sub-tick resolution.

    The simulator advances every unit once per (coarse) tick.  At BVR closing
    speeds a single 1 s step moves a missile well over a kilometre, so a hit can
    be jumped over or a hard-turning endgame can be mis-resolved by a straight
    chord.  For each engaged (missile, target) pair this class rewinds the pair
    to its begin-of-tick state and re-integrates ONLY that pair's flight physics
    in small substeps (a lightweight, pair-local simulation - no radar, guidance
    recompute, or other units), running a continuous-collision check on every
    substep.  On geometric contact it defers the kill/no-kill decision to the
    CCD manager's probabilistic ``on_hit`` callback.

    The substeps are non-authoritative: the pair is restored to its real
    end-of-tick state afterwards, so the substepper only emits hit events.
    """

    def __init__(self, config: SubstepConfig | None = None) -> None:
        self.cfg = config or SubstepConfig()

    def run_tick_with_substeps(self, sim, tick_secs: float):
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
            # Resolve the target the missile is actually guiding on. Real missiles
            # carry no `locked_target` object attribute — the lock lives on the
            # radar as a unit ID — so resolve that ID to a unit via sim as the
            # authoritative source, falling back to the missile's target object.
            tgt = (
                getattr(u, "locked_target", None)
                or self._resolve_radar_lock_target(u, sim)
                or getattr(u, "target", None)
            )
            if tgt is None or getattr(tgt, "is_destroyed", False):
                continue
            try:
                if units_distance_km(u, tgt) <= self.cfg.engage_distance_km:
                    out.append((u, tgt))
            except Exception:
                continue
        return out

    def _substep_pair(self, sim, missile, target, tick_secs: float, calc) -> None:
        """Run continuous-collision detection along the missile's *authoritative*
        sub-stepped path for this tick.

        The missile records the fine positions it actually flew (``tick_path``)
        while integrating its own guidance + movement. We walk those segments
        against the target's position (linearly interpolated between its begin-
        and end-of-tick poses — the target is slow and gently turning), so the
        miss distance reflects the real trajectory. The first geometric contact
        is handed to ``ccd.on_hit`` with that miss distance (range-dependent Pk).
        """
        ccd = getattr(sim, "ccd", None)

        # Arming gate: an unarmed fuze cannot detonate yet — the missile flies on.
        arm_time = getattr(missile, "arming_time_s", 0.0)
        if float(getattr(missile, "elapsed_time_s", 0.0)) < float(arm_time):
            return

        path = getattr(missile, "tick_path", None)
        if not path or len(path) < 2:
            # No recorded sub-path (e.g. a non-Missile stub): single-segment CCD.
            self._single_segment_fallback(sim, missile, target, tick_secs, calc)
            return

        t_begin = self._begin_pose(ccd, target) or self._pose(target)
        t_end = self._pose(target)

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
            t0 = self._lerp_pose(t_begin, t_end, f0)
            t1 = self._lerp_pose(t_begin, t_end, f1)
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
            t_cpa = self._lerp_pose(
                self._lerp_pose(t_begin, t_end, f0),
                self._lerp_pose(t_begin, t_end, f1),
                best_ts,
            )
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
                except Exception:
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
