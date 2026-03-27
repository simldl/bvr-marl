from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from typing import Optional

from air_to_air_rl.simulator.core.helpers import clamp, units_distance_km


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
    """Executes substeps only for (Missile <-> Target) pairs after the normal tick.
    Does not modify the final tick state: only emits events."""

    def __init__(self, config: SubstepConfig | None = None) -> None:
        self.cfg = config or SubstepConfig()

    def run_tick_with_substeps(self, sim, tick_secs: float):
        pairs = self._collect_candidates(sim.active_units.values())
        if not pairs:
            return []

        ccd = getattr(sim, "ccd", None)
        calc = getattr(ccd, "hit_calc", None) if ccd is not None else None
        if calc is None:
            calc = getattr(sim, "hit_calc", None)

        if calc is not None:
            for missile, target in pairs:
                hit, victim, t_frac = calc.check_and_maybe_hit(
                    missile=missile, potential_targets=[target], tick_secs=tick_secs
                )
                if hit and victim is not None:
                    if hasattr(sim, "ccd") and hasattr(sim.ccd, "on_hit"):
                        sim.ccd.on_hit(missile, victim, t_frac or 0.0, sim)
        else:
            for missile, target in pairs:
                self._integrate_temporarily_with_restore(sim, missile, target, tick_secs)

        return []

    def _collect_candidates(self, units: Iterable[object]) -> list[tuple[object, object]]:
        out: list[tuple[object, object]] = []
        for u in units:
            if not self._is_missile(u):
                continue
            tgt = getattr(u, "locked_target", None) or getattr(u, "target", None)
            if tgt is None or getattr(tgt, "is_destroyed", False):
                continue
            try:
                if units_distance_km(u, tgt) <= self.cfg.engage_distance_km:
                    out.append((u, tgt))
            except Exception:
                continue
        return out

    def _integrate_temporarily_with_restore(self, sim, missile, target, tick_secs: float) -> None:
        """Substeps on begin-of-tick states; then restore to final state."""
        ccd = getattr(sim, "ccd", None)
        has_hooks = ccd and all(
            hasattr(ccd, name) for name in ("capture_state", "get_begin_state", "apply_state")
        )
        if not has_hooks:
            return

        cur_m = ccd.capture_state(missile)
        cur_t = ccd.capture_state(target)

        begin_m = ccd.get_begin_state(missile)
        begin_t = ccd.get_begin_state(target)
        if begin_m is None or begin_t is None:
            return
        ccd.apply_state(missile, begin_m)
        ccd.apply_state(target, begin_t)

        n_sub = self._estimate_substeps(missile, target, tick_secs)
        sub_dt = max(tick_secs / max(1, n_sub), self.cfg.min_dt)
        calc = getattr(sim, "ccd", None)
        calc = getattr(calc, "hit_calc", None) if calc is not None else None
        calc = calc or getattr(sim, "hit_calc", None)

        t_acc = 0.0
        while t_acc < tick_secs - 1e-12:
            dt = min(sub_dt, tick_secs - t_acc)

            if self.cfg.physics_only:
                self._physics_only_step(sim, missile, dt)
                self._physics_only_step(sim, target, dt)
            else:
                step_m = getattr(missile, "substep_update", None) or self._physics_only_step
                step_t = getattr(target, "substep_update", None) or self._physics_only_step
                if step_m is self._physics_only_step:
                    step_m(sim, missile, dt)
                else:
                    step_m(dt, sim)
                if step_t is self._physics_only_step:
                    step_t(sim, target, dt)
                else:
                    step_t(dt, sim)

            if calc is not None:
                hit, victim, t_frac = calc.check_and_maybe_hit(
                    missile=missile, potential_targets=[target], tick_secs=dt
                )
                if (
                    hit
                    and victim is not None
                    and hasattr(sim, "ccd")
                    and hasattr(sim.ccd, "on_hit")
                ):
                    sim.ccd.on_hit(missile, victim, t_frac or 0.0, sim)
                    break

            t_acc += dt
        ccd.apply_state(missile, cur_m)
        ccd.apply_state(target, cur_t)

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
