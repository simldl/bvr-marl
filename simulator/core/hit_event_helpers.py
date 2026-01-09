from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Callable, Any, Dict

from simulator.core.helpers import units_distance_km
from simulator.core.events import MissileEngagementEvent, UnitDestroyedEvent
from simulator.core.hit_calculation import MissileHitCalculator, HitParams


# =============================== Config =====================================

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


# ============================== Callbacks ===================================

def default_on_hit(missile: Any, target: Any, t_frac: float, sim: Any) -> None:
    """
    Standard hit handling: destroy target and missile and log events.
    """
    # Remove target from simulation if active
    if target.id in sim.active_units:
        target_unit = sim.remove_unit(target.id)
        # Remove any countermeasures deployed by the target
        if target_unit and hasattr(sim, '_remove_countermeasures_for_parent'):
            sim._remove_countermeasures_for_parent(target_unit)
    # Always log target destruction event
    sim.log_event(UnitDestroyedEvent(sim, missile, target))

    # Remove missile from simulation if active
    if missile.id in sim.active_units:
        sim.remove_unit(missile.id)
    # Always log missile self-destruction event
    sim.log_event(UnitDestroyedEvent(sim, missile, missile))


def stochastic_on_hit(missile: Any, target: Any, t_frac: float, sim: Any) -> None:
    """
    Stochastic hit handling: uses missile.hit_probability to determine if target is destroyed.
    Missile is always destroyed on proximity hit, but target destruction is probabilistic.
    """
    import random

    # Get hit probability from missile configuration (default to 0.85 if not set)
    hit_prob = getattr(missile, 'hit_probability', 0.85)

    # Roll for hit success
    if random.random() < hit_prob:
        # Successful hit - destroy target
        if target.id in sim.active_units:
            target_unit = sim.remove_unit(target.id)
            # Remove any countermeasures deployed by the target
            if target_unit and hasattr(sim, '_remove_countermeasures_for_parent'):
                sim._remove_countermeasures_for_parent(target_unit)
        # Log target destruction event
        sim.log_event(UnitDestroyedEvent(sim, missile, target))
    else:
        # Miss - target survives, but we still log an engagement event
        # (optional: could add MissileMissEvent if desired)
        pass

    # Missile is always destroyed when reaching proximity fuze distance
    if missile.id in sim.active_units:
        sim.remove_unit(missile.id)
    # Log missile self-destruction event
    sim.log_event(UnitDestroyedEvent(sim, missile, missile))


def default_target_selector(missile: Any) -> Optional[Any]:
    """
    Selects the primary target of the missile.
    Prefers locked_target, otherwise uses target.
    """
    return getattr(missile, "locked_target", None) or getattr(missile, "target", None)


# ============================ CCD-Manager ===================================

class MissileCCDManager:
    """
    Centralizes snapshot, 10-km gate, CCD and event emission.

    Typical flow per tick (called by simulator):
        1) begin_tick_snapshot(units)
        2) (Units update/move)
        3) post_update_ccd(units, tick_secs, sim)

    Additionally provides state hooks that substepping code can use
    for temporary integration:
        - capture_state(unit)      -> Dict
        - get_begin_state(unit)    -> Dict | None
        - apply_state(unit, state) -> None
    """

    # ------------------------------ Init ------------------------------------

    def __init__(
        self,
        config: CCDConfig = CCDConfig(),
        hit_params: Optional[HitParams] = None,
        on_hit: Callable[[Any, Any, float, Any], None] = default_on_hit,
        target_selector: Callable[[Any], Optional[Any]] = default_target_selector,
    ) -> None:
        self.cfg = config
        self.on_hit = on_hit
        self.target_selector = target_selector
        self.hit_calc = MissileHitCalculator(
            hit_params or HitParams(
                fuse_radius_m=self.cfg.fuse_radius_m,
                target_radius_m=self.cfg.target_radius_m,
                max_consider_range_m=self.cfg.max_consider_range_m,
            )
        )

        self._engaged_once: set[int] = set()
        self._begin_states: Dict[int, Dict[str, Any]] = {}

    # --------------------------- Lifecycle -----------------------------------

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
        # Create a copy to avoid "dictionary changed during iteration" error
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

    # ------------------------- Snapshot / Restore ----------------------------

    def capture_state(self, u: Any) -> Dict[str, Any]:
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
            "yaw":   f("yaw_deg"),
            "pitch": f("pitch_deg"),
            "roll":  f("roll_deg"),
            "speed": f("speed"),
        }
        if hasattr(u, "desired_yaw_deg"):
            state["dyaw"] = f("desired_yaw_deg", f("yaw_deg", 0.0))
        if hasattr(u, "desired_pitch_deg"):
            state["dpitch"] = f("desired_pitch_deg", f("pitch_deg", 0.0))
        return state

    def get_begin_state(self, u: Any) -> Optional[Dict[str, Any]]:
        """
        Get the begin-of-tick state of a unit (if available).
        """
        return self._begin_states.get(self._uid(u))

    @staticmethod
    def apply_state(u: Any, st: Optional[Dict[str, Any]]) -> None:
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
        if "yaw"   in st and hasattr(u, "yaw_deg"):   u.yaw_deg   = st["yaw"]
        if "pitch" in st and hasattr(u, "pitch_deg"): u.pitch_deg = st["pitch"]
        if "roll"  in st and hasattr(u, "roll_deg"):  u.roll_deg  = st["roll"]
        if "speed" in st and hasattr(u, "speed"):     u.speed     = st["speed"]
        if "dyaw"   in st and hasattr(u, "desired_yaw_deg"):   u.desired_yaw_deg   = st["dyaw"]
        if "dpitch" in st and hasattr(u, "desired_pitch_deg"): u.desired_pitch_deg = st["dpitch"]

    # ----------------------------- Utilities --------------------------------

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

    # ---------------------------- Maintenance --------------------------------

    def reset_engagement_tracking(self) -> None:
        """
        Optional helper for tests/resets:
        clears the 'once per missile' engagement markers.
        """
        self._engaged_once.clear()