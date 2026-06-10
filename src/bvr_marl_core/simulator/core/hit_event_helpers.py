from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Optional

from bvr_marl_core.simulator.core.events import MissileEngagementEvent, UnitDestroyedEvent
from bvr_marl_core.simulator.core.helpers import units_distance_km
from bvr_marl_core.simulator.core.hit_calculation import HitParams, MissileHitCalculator


def kill_probability(missile: Any, miss_distance_m: float | None) -> float:
    """Probability that a proximity detonation destroys the target.

    Combines the missile's center lethality (``hit_probability``) with a
    Gaussian fall-off in the closest-approach miss distance, scaled by the
    missile's ``lethal_radius_m``:

        Pk(d) = hit_probability * exp(-(d / lethal_radius_m)^2)

    A missile with ``lethal_radius_m <= 0`` (the default) or a caller that does
    not supply a miss distance keeps the flat ``hit_probability`` — preserving
    legacy behaviour. Operational missiles set ``lethal_radius_m`` so that a
    target which opens the terminal miss distance (by breaking lock or a hard
    last-ditch break) can survive a proximity detonation.
    """
    base = float(getattr(missile, "hit_probability", 0.85))
    if miss_distance_m is None:
        return base
    lethal_r = float(getattr(missile, "lethal_radius_m", 0.0))
    if lethal_r <= 0.0:
        return base
    d = max(0.0, float(miss_distance_m))
    return base * math.exp(-((d / lethal_r) ** 2))


@dataclass
class CCDConfig:
    """
    Settings for Continuous Collision Detection (CCD).

    - fuse_radius_m:        Proximity/fuze radius of the missile.
    - target_radius_m:      Optional target radius (0.0 = point target).
    - max_consider_range_m: Coarse pre-filter in meters (e.g., 50 km).
    - within_lock_gate_m:   Distance at which an engagement event is logged
                            and CCD becomes active (e.g., 10 km).
    """

    fuse_radius_m: float = 500.0
    target_radius_m: float = 0.0
    max_consider_range_m: float = 50_000.0
    within_lock_gate_m: float = 10_000.0


def default_on_hit(
    missile: Any, target: Any, t_frac: float, sim: Any, miss_distance_m: float | None = None
) -> None:
    """Standard hit handling: destroy target and missile and log events."""
    if target.id in sim.active_units:
        target_unit = sim.remove_unit(target.id)
        if target_unit and hasattr(sim, "_remove_countermeasures_for_parent"):
            sim._remove_countermeasures_for_parent(target_unit)
    sim.log_event(UnitDestroyedEvent(sim, missile, target))

    if missile.id in sim.active_units:
        sim.remove_unit(missile.id)
    sim.log_event(UnitDestroyedEvent(sim, missile, missile))


def stochastic_on_hit(
    missile: Any, target: Any, t_frac: float, sim: Any, miss_distance_m: float | None = None
) -> None:
    """
    Stochastic hit handling: target destruction is probabilistic.
    The missile is always destroyed on a proximity detonation, but the target
    is destroyed only if the (range-dependent) kill-probability roll succeeds.
    When ``miss_distance_m`` is supplied, Pk falls off with the closest-approach
    distance (see :func:`kill_probability`); otherwise the flat
    ``hit_probability`` is used. Uses sim.rnd_gen for reproducibility.
    """
    hit_prob = kill_probability(missile, miss_distance_m)
    rnd = getattr(sim, "rnd_gen", None)
    roll = rnd.random() if rnd is not None else __import__("random").random()

    if roll < hit_prob:
        if target.id in sim.active_units:
            target_unit = sim.remove_unit(target.id)
            if target_unit and hasattr(sim, "_remove_countermeasures_for_parent"):
                sim._remove_countermeasures_for_parent(target_unit)
        sim.log_event(UnitDestroyedEvent(sim, missile, target))

    if missile.id in sim.active_units:
        sim.remove_unit(missile.id)
    sim.log_event(UnitDestroyedEvent(sim, missile, missile))


def default_target_selector(missile: Any) -> Any | None:
    """
    Selects the primary target of the missile.
    Prefers locked_target, otherwise uses target.
    """
    return getattr(missile, "locked_target", None) or getattr(missile, "target", None)


class MissileCCDManager:
    """
    Centralizes snapshot, 10-km gate, CCD and event emission.

    Typical flow per tick (called by simulator):
        1) begin_tick_snapshot(units)
        2) (Units update/move)
        3) post_update_ccd(units, tick_secs, sim)

    Additionally provides state hooks that substepping code can use
    for temporary integration:
        - capture_state(unit)      -> dict
        - get_begin_state(unit)    -> dict | None
        - apply_state(unit, state) -> None
    """

    def __init__(
        self,
        config: CCDConfig = CCDConfig(),
        hit_params: HitParams | None = None,
        on_hit: Callable[[Any, Any, float, Any], None] = default_on_hit,
        target_selector: Callable[[Any], Any | None] = default_target_selector,
    ) -> None:
        self.cfg = config
        self.on_hit = on_hit
        self.target_selector = target_selector
        self.hit_calc = MissileHitCalculator(
            hit_params
            or HitParams(
                fuse_radius_m=self.cfg.fuse_radius_m,
                target_radius_m=self.cfg.target_radius_m,
                max_consider_range_m=self.cfg.max_consider_range_m,
            )
        )

        self._engaged_once: set[int] = set()
        self._begin_states: dict[int, dict[str, Any]] = {}

    def begin_tick_snapshot(self, units: Iterable[Any]) -> None:
        """
        At the beginning of the tick, create CCD snapshot and buffer
        begin-of-tick states (for substepping restore).
        """
        self._begin_states.clear()
        self.hit_calc.begin_tick_snapshot(units)

        for u in units:
            if not hasattr(u, "position"):
                continue
            self._begin_states[self._uid(u)] = self.capture_state(u)

    def post_update_ccd(self, units: Iterable[Any], tick_secs: float, sim: Any) -> None:
        """
        After the movement/update step, execute CCD for relevant
        missile-target pairs.
        """
        units_list = list(units)
        for missile in units_list:
            if not self._is_active_missile(missile):
                continue

            target = self.target_selector(missile)
            if target is None or getattr(target, "is_destroyed", False):
                continue

            dist_km = units_distance_km(missile, target)
            if dist_km * 1000.0 > self.hit_calc.params.max_consider_range_m:
                continue

            if dist_km * 1000.0 <= self.cfg.within_lock_gate_m:
                uid = self._uid(missile)
                if uid not in self._engaged_once:
                    self._engaged_once.add(uid)
                    sim.log_event(MissileEngagementEvent(sim, missile, target))
            else:
                continue
            hit, victim, t_frac = self.hit_calc.check_and_maybe_hit(
                missile=missile,
                potential_targets=[target],
                tick_secs=tick_secs,
            )
            if hit and victim is not None:
                self.on_hit(missile, victim, t_frac or 0.0, sim)

    def capture_state(self, u: Any) -> dict[str, Any]:
        """
        Minimal state needed for substeps:
        Position + (Yaw/Pitch/Roll), Speed and desired control values.
        """
        pos = getattr(u, "position", None)
        pos_copy = pos.copy() if hasattr(pos, "copy") else pos

        def f(name: str, default: float = 0.0) -> float:
            return float(getattr(u, name, default))

        state = {
            "pos": pos_copy,
            "yaw": f("yaw_deg"),
            "pitch": f("pitch_deg"),
            "roll": f("roll_deg"),
            "speed": f("speed"),
        }
        if hasattr(u, "desired_yaw_deg"):
            state["dyaw"] = f("desired_yaw_deg", f("yaw_deg", 0.0))
        if hasattr(u, "desired_pitch_deg"):
            state["dpitch"] = f("desired_pitch_deg", f("pitch_deg", 0.0))
        return state

    def get_begin_state(self, u: Any) -> dict[str, Any] | None:
        """
        Get the begin-of-tick state of a unit (if available).
        """
        return self._begin_states.get(self._uid(u))

    @staticmethod
    def apply_state(u: Any, st: dict[str, Any] | None) -> None:
        """
        Restore a unit to a previously captured state.
        Missing fields are silently ignored.
        """
        if not st:
            return

        p = st.get("pos")
        if p is not None and hasattr(u, "position"):
            u.position.lat = getattr(p, "lat", getattr(u.position, "lat", 0.0))
            u.position.lon = getattr(p, "lon", getattr(u.position, "lon", 0.0))
            u.position.alt = getattr(p, "alt", getattr(u.position, "alt", 0.0))
        if "yaw" in st and hasattr(u, "yaw_deg"):
            u.yaw_deg = st["yaw"]
        if "pitch" in st and hasattr(u, "pitch_deg"):
            u.pitch_deg = st["pitch"]
        if "roll" in st and hasattr(u, "roll_deg"):
            u.roll_deg = st["roll"]
        if "speed" in st and hasattr(u, "speed"):
            u.speed = st["speed"]
        if "dyaw" in st and hasattr(u, "desired_yaw_deg"):
            u.desired_yaw_deg = st["dyaw"]
        if "dpitch" in st and hasattr(u, "desired_pitch_deg"):
            u.desired_pitch_deg = st["dpitch"]

    @staticmethod
    def _uid(u: Any) -> int:
        """
        Robust key: use .id if available, otherwise id(u).
        """
        return getattr(u, "id", id(u))

    @staticmethod
    def _is_active_missile(u: Any) -> bool:
        """
        Allows multiple "missile" indicators for compatibility
        (flag, type name or unit_type).
        """
        if getattr(u, "is_destroyed", False):
            return False
        if getattr(u, "is_missile", False):
            return True
        if getattr(u, "unit_type", "").lower() == "missile":
            return True
        if type(u).__name__.lower().endswith("missile"):
            return True
        return False

    def reset_engagement_tracking(self) -> None:
        """
        Optional helper for tests/resets:
        clears the 'once per missile' engagement markers.
        """
        self._engaged_once.clear()
