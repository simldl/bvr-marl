from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from bvr_marl_core.simulator.core.effectiveness import KillProbabilityModel
from bvr_marl_core.simulator.core.events import (
    AircraftMortallyHitEvent,
    MissileDetonatedEvent,
    MissileEngagementEvent,
    MissileEnteredTerminalRegionEvent,
    MissileFuzeTriggeredEvent,
    MissileTerminalEvent,
    UnitDestroyedEvent,
)
from bvr_marl_core.simulator.core.helpers import units_distance_km
from bvr_marl_core.simulator.core.hit_calculation import HitParams, MissileHitCalculator

# Shared, stateless kill-probability pipeline. The submodels carry no per-call
# state, so a single module-level instance is reused for every detonation.
_KILL_MODEL = KillProbabilityModel()

# Stochastic kill delay: a lethal hit does not remove the target instantly. The
# aircraft is mortally hit, spirals out of the fight, and dies after a random lag
# ~ Exponential(mean) clamped to a cap. The kill is credited when the death fires.
KILL_DELAY_MEAN_S = 3.0
KILL_DELAY_MAX_S = 10.0


def mark_mortally_hit(target: Any, missile: Any, sim: Any) -> None:
    """Flag ``target`` as lethally hit and schedule its (delayed) death.

    Records the killer for attribution and a random death time; the target's own
    update fires the ``UnitDestroyedEvent`` and removal when that time arrives.
    Idempotent — a second lethal hit on an already-dying target is ignored.
    """
    if getattr(target, "is_mortally_hit", False):
        return
    streams = getattr(sim, "random_streams", None)
    if streams is None:
        rnd = getattr(sim, "rnd_gen", None)
        expovariate = getattr(rnd, "expovariate", None)
        delay = (
            min(KILL_DELAY_MAX_S, expovariate(1.0 / KILL_DELAY_MEAN_S))
            if callable(expovariate)
            else KILL_DELAY_MEAN_S
        )
    else:
        rng = streams.generator("kill_delay", getattr(target, "id", 0))
        delay = min(KILL_DELAY_MAX_S, float(rng.exponential(KILL_DELAY_MEAN_S)))
    now = float(getattr(sim, "elapsed_time_s", 0.0))
    target.is_mortally_hit = True
    target._death_killer = missile
    target._death_time_s = now + delay
    logger = getattr(sim, "log_event", None)
    if callable(logger):
        logger(AircraftMortallyHitEvent(sim, target, missile, target._death_time_s))
    # Out of the fight immediately: stop radiating (weapons/sensors are gated in
    # their own paths, and the airframe enters a death spiral in movement).
    try:
        target.radar_emitting = False
    except (AttributeError, TypeError):
        pass


def kill_probability_components(
    missile: Any, target: Any | None, miss_distance_m: float | None
) -> tuple[float, dict[str, float]]:
    """Return ``(Pk, components)`` from the decomposed kill-probability model.

    ``components`` is the flat ``P_int / P_fuze / P_wh / P_vul / P_trk``
    breakdown logged on the terminal event.
    """
    return _KILL_MODEL.compute(missile, target, miss_distance_m)


def kill_probability(missile: Any, miss_distance_m: float | None) -> float:
    """Probability that a proximity detonation destroys the target.

    Delegates to the decomposed :class:`KillProbabilityModel`, which factors Pk
    into intercept, fuze, warhead, vulnerability, and terminal-tracking terms:

        Pk = P_int * P_fuze * P_wh * P_vul * P_trk

    For a geometrically valid intercept against a neutral target/track, this is
    ``Pk(d) = P_fuze * exp(-(d / lethal_radius_m)^2)``. Guidance quality has
    already manifested in the achieved CPA and is not multiplied again. The
    legacy ``hit_probability`` remains compatibility and shot-quality metadata;
    the remaining terms are 1.0 until populated (Phases 5–7). A non-positive
    ``lethal_radius_m`` or a missing miss distance keeps the flat lethality.
    """
    pk, _ = _KILL_MODEL.compute(missile, None, miss_distance_m)
    return pk


def _closing_speed_mps(missile: Any, target: Any) -> float | None:
    """Closing speed (m/s, positive = closing) along the missile→target LOS.

    Best-effort: uses each unit's ``velocity`` (ENU: vx=East, vy=North, vz=Up)
    and a flat-earth LOS from lat/lon/alt deltas. Returns ``None`` if anything
    needed is missing. Cheap — called once per detonation.
    """
    try:
        mp, tp = missile.position, target.position
        mv, tv = missile.velocity, target.velocity
        cos_lat = math.cos(math.radians(mp.lat))
        de = (tp.lon - mp.lon) * 111_000.0 * cos_lat
        dn = (tp.lat - mp.lat) * 111_000.0
        du = tp.alt - mp.alt
        rng = math.sqrt(de * de + dn * dn + du * du)
        if rng < 1e-6:
            return None
        ex, ey, eu = de / rng, dn / rng, du / rng
        # Closing = component of (v_missile - v_target) along the LOS to target.
        return (mv.vx - tv.vx) * ex + (mv.vy - tv.vy) * ey + (mv.vz - tv.vz) * eu
    except Exception:
        return None


def build_terminal_record(
    missile: Any,
    target: Any,
    miss_distance_m: float | None,
    pk: float,
    roll: float,
    killed: bool,
    sim: Any,
    components: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Assemble the flat terminal-event record for one detonation.

    Reads only state already resident on the missile/target/sim (launch context
    cached at fire time in ``missile._launch_context``, CPA geometry in
    ``missile._last_cpa_event``, seeker/track state on the radar and target
    provider). All access is guarded so logging can never break the sim.
    """
    radar = getattr(missile, "radar", None)
    provider = getattr(missile, "target_provider", None)

    def _attr(obj, name, default=None):
        try:
            return getattr(obj, name, default)
        except Exception:
            return default

    terminal_lock = None
    seeker_stable = None
    if radar is not None:
        try:
            terminal_lock = radar.get_locked_target() is not None
        except Exception:
            terminal_lock = None
        if hasattr(radar, "has_stable_seeker_track"):
            try:
                seeker_stable = bool(radar.has_stable_seeker_track())
            except Exception:
                seeker_stable = None

    track_age_s = _attr(provider, "last_confirmed_track_age_s")
    has_fresh_track = None
    if provider is not None and hasattr(provider, "has_fresh_track"):
        try:
            has_fresh_track = bool(provider.has_fresh_track())
        except Exception:
            has_fresh_track = None

    cpa = _attr(missile, "_last_cpa_event") or {}

    record: dict[str, Any] = {
        "time_s": _attr(sim, "elapsed_time_s"),
        "shooter_id": _attr(getattr(missile, "source", None), "id"),
        "target_id": _attr(target, "id"),
        "missile_id": _attr(missile, "id"),
        "missile_type": _attr(missile, "name") or type(missile).__name__,
        # Terminal seeker / track quality
        "terminal_lock": terminal_lock,
        "terminal_seeker_stable": seeker_stable,
        "terminal_track_age_s": track_age_s,
        "terminal_has_fresh_track": has_fresh_track,
        # Closest-approach geometry
        "miss_distance_m": miss_distance_m,
        "closing_mps_at_cpa": _closing_speed_mps(missile, target),
        "cpa_time_s": cpa.get("time_s"),
        "cpa_t_frac": cpa.get("t_frac"),
        "cpa_miss_m": cpa.get("miss_m"),
        # Fuze / kill resolution
        "fuze_armed": float(_attr(missile, "elapsed_time_s", 0.0) or 0.0)
        >= float(_attr(missile, "arming_time_s", 0.0) or 0.0),
        "fuze_decision": "detonate",
        "pk": pk,
        "kill_roll": roll,
        "killed": killed,
    }
    # Three distinct outcomes, reported separately because they answer different
    # questions and collapsing them hides where shots are actually being lost:
    #   detonated     - the fuze fired at all (the weapon reached its target)
    #   geometric_hit - closest approach fell inside the warhead's lethal radius
    #   killed        - the Pk roll then succeeded
    # A wide fuze radius relative to the lethal radius produces many detonations
    # that were never geometric hits, each drawing a low Pk; without the split
    # that reads as bad luck rather than as shots arriving too far out.
    lethal_radius_m = _attr(missile, "lethal_radius_m")
    record["detonated"] = True
    record["lethal_radius_m"] = lethal_radius_m
    record["fuze_radius_m"] = _attr(getattr(_attr(sim, "ccd"), "cfg", None), "fuse_radius_m")
    record["geometric_hit"] = (
        bool(miss_distance_m is not None and lethal_radius_m and miss_distance_m <= lethal_radius_m)
        if miss_distance_m is not None
        else None
    )
    # Kill-probability factorization (populated once the decomposed model lands).
    record["pk_components"] = dict(components) if components else {}
    # Launch context cached at fire time (flat-merged with a launch_ prefix kept
    # by the producer); absent in environments that fire without a weapon system.
    launch_ctx = _attr(missile, "_launch_context")
    if isinstance(launch_ctx, dict):
        for k, v in launch_ctx.items():
            record.setdefault(k if k.startswith("launch_") else f"launch_{k}", v)
    return record


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
    hit_prob, components = kill_probability_components(missile, target, miss_distance_m)
    streams = getattr(sim, "random_streams", None)
    if streams is not None:
        stream_id = f"{getattr(missile, 'id', 0)}:{getattr(target, 'id', 0)}"
        roll = float(streams.generator("terminal_lethality", stream_id).random())
    else:
        rnd = getattr(sim, "rnd_gen", None)
        roll = float(rnd.random()) if rnd is not None else 0.5
    killed = roll < hit_prob

    sim.log_event(MissileFuzeTriggeredEvent(sim, missile, target, miss_distance_m))
    terminal_record = build_terminal_record(
        missile, target, miss_distance_m, hit_prob, roll, killed, sim, components
    )
    sim.log_event(MissileDetonatedEvent(sim, missile, target, terminal_record))
    # Compatibility record retained for telemetry consumers during migration.
    sim.log_event(
        MissileTerminalEvent(
            sim,
            missile,
            target,
            terminal_record,
        )
    )

    if killed:
        # Stochastic time offset: the target is mortally hit now but dies after a
        # random lag (death spiral). The UnitDestroyedEvent (kill credit) and the
        # removal are deferred to the target's own update at the scheduled time.
        mark_mortally_hit(target, missile, sim)

    # The missile always detonates and is removed this tick.
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
        config: CCDConfig | None = None,
        hit_params: HitParams | None = None,
        on_hit: Callable[[Any, Any, float, Any], None] = default_on_hit,
        target_selector: Callable[[Any], Any | None] = default_target_selector,
    ) -> None:
        self.cfg = config or CCDConfig()
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

            target = None
            operational_track = getattr(missile, "weapon_track", None) is not None
            resolver = getattr(sim, "evaluator_target_for_weapon", None)
            if isinstance(getattr(sim, "_weapon_truth_associations", None), dict) and callable(
                resolver
            ):
                target = resolver(missile)
            if target is None and operational_track:
                record = getattr(sim, "record_diagnostic", None)
                # A weapon whose attributed aircraft has already left the fight has
                # nothing left to resolve against. Terminating it here keeps it from
                # coasting to lifetime expiry as an unattributable weapon, which both
                # wastes the flight and buries the genuine attribution failures this
                # diagnostic exists to surface.
                departed = getattr(sim, "evaluator_weapon_target_departed", None)
                if callable(departed) and departed(missile):
                    missile.should_be_removed = True
                    missile.removal_reason = "target_destroyed"
                    if callable(record):
                        record("weapon_target_departed")
                elif callable(record):
                    record("missing_evaluator_target")
                continue
            if target is None:
                target = self.target_selector(missile)
            if target is None or getattr(target, "is_destroyed", False):
                continue
            # Backstop for the scenario's rules of engagement. Launch vetoes are the
            # first line, but a weapon can be re-attributed in flight, so a protected
            # unit must never be resolvable as a victim here either.
            if getattr(target, "is_non_engageable", False):
                record = getattr(sim, "record_diagnostic", None)
                if callable(record):
                    record("hit_suppressed_non_engageable")
                continue
            # Already mortally hit and dying — do not resolve a second lethal hit.
            if getattr(target, "is_mortally_hit", False):
                continue

            dist_km = units_distance_km(missile, target)
            if dist_km * 1000.0 > self.hit_calc.params.max_consider_range_m:
                continue

            if dist_km * 1000.0 <= self.cfg.within_lock_gate_m:
                uid = self._uid(missile)
                if uid not in self._engaged_once:
                    self._engaged_once.add(uid)
                    sim.log_event(
                        MissileEnteredTerminalRegionEvent(sim, missile, target, dist_km * 1000.0)
                    )
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
