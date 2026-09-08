"""The single definition of "may this aircraft launch right now?".

Before this module the question was answered in three independent places that
disagreed with each other, and every disagreement showed up as a metric that could
not be reconciled with the behaviour it was supposed to describe:

1. ``ObservationHelper.get_fire_feasibility`` produced the observation's ``can_fire``
   bit (own_state index :data:`OWN_IDX_CAN_FIRE`), which is also what the fire-gradient
   mask keys off. Its docstring promised "lock, FOV, gimbal, range, inventory,
   cooldown" but it never looked at the cooldown -- ``get_weapon_status`` hardcoded
   ``missile_cooldown_left_s = 0.0`` -- and it never looked at per-target saturation.
   So the mask permitted trigger presses during the 2 s post-launch cooldown, every one
   of which was vetoed on the launch path, and ``tactical/trigger_precision_rate``
   read ~0.000 while the mask was nominally "only allowing feasible presses".

2. ``WeaponFiringHandler`` computed ``shot_opportunity_this_step`` from cooldown,
   inventory, saturation and the aircraft's OWN radar lock -- no datalink, no gimbal,
   no range. Once a scenario includes an AWACS, a datalink-cued shot is a real shot the
   launch path will take, so the opportunity counter undercounted exactly the
   scenario the AWACS was added to create.

3. The launch path itself (``WeaponSystem.fire_missile*``) applied its own gates and
   reported the outcome through ``diagnostics``.

:func:`evaluate_fire_gates` is now the one computation. ``can_fire`` and
``shot_opportunity_this_step`` are the *same field of the same object*, so they cannot
drift again; the launch path stays authoritative for the actual shot, and any
divergence between it and this prediction is a real finding rather than a bookkeeping
artefact.

Note on ``radar_lock`` vs ``has_lock``: ``tactical/lock_rate`` deliberately reports the
aircraft's own-radar lock, because it measures that aircraft's sensor discipline. Firing
feasibility deliberately accepts a datalink-cued lock, because the launch path does.
Both live here so the distinction is stated once, on purpose, instead of arising from
two files having drifted.
"""

from __future__ import annotations

from dataclasses import dataclass

from bvr_marl_core.domain.tactical_contact import TacticalContact

# Fraction of the radar's rated range treated as usable for a launch decision. Matches
# the long-standing value in the observation path; kept as a named constant so the two
# call sites cannot pick different margins.
RADAR_RANGE_USABLE_FRACTION = 0.95


@dataclass(frozen=True)
class FireGates:
    """Every launch gate evaluated against one target at one instant."""

    inventory_ok: bool = False
    radar_lock: bool = False
    datalink_lock: bool = False
    target_in_fov: bool = False
    gimbal_ok: bool = False
    radar_range_ok: bool = False
    weapon_range_ok: bool = True
    cooldown_ok: bool = False
    target_not_saturated: bool = False
    # Rules of engagement: the scenario may forbid shooting this target at all
    # (AWACS and other support assets). Defaults True so a FireGates built without
    # the rule -- test doubles, callers with no simulator -- keeps prior behaviour.
    roe_ok: bool = True
    remaining_missiles: int = 0
    error: str | None = None

    @property
    def has_lock(self) -> bool:
        """Any lock the launch path will accept, own radar or datalink-cued."""
        return self.radar_lock or self.datalink_lock

    @property
    def launch_range_ok(self) -> bool:
        """Is the target inside a range from which this launch could actually work?

        Two independent ceilings, composed in the only order that is physically
        meaningful. A datalink shot may be launched beyond the shooter's own RADAR
        range -- someone else is holding the track -- but nothing relaxes the
        MISSILE's kinematic reach, so ``weapon_range_ok`` is an unconditional
        conjunct.

        The weapon ceiling was previously absent: the gate was radar range alone, at
        95% of rated. That makes the gate vacuous behind a long-range radar -- an
        agent can hold station far outside any viable shot with ``wasted_range``
        reading 0.00, because no shot from there was ever kinematically possible.
        """
        return (self.radar_range_ok or self.datalink_lock) and self.weapon_range_ok

    @property
    def can_fire(self) -> bool:
        return (
            self.inventory_ok
            and self.has_lock
            and self.target_in_fov
            and self.gimbal_ok
            and self.launch_range_ok
            and self.cooldown_ok
            and self.target_not_saturated
            and self.roe_ok
        )

    @property
    def veto_reason(self) -> str | None:
        """First unmet gate, ordered so the reported reason is the actionable one."""
        if self.error is not None:
            return self.error
        if not self.inventory_ok:
            return "winchester"
        if not self.has_lock:
            return "no_radar_or_datalink_lock"
        if not self.target_in_fov:
            return "not_in_fov"
        if not self.gimbal_ok:
            return "outside_gimbal_limits"
        if not self.weapon_range_ok:
            return "outside_missile_launch_range"
        if not self.launch_range_ok:
            return "outside_own_radar_range"
        if not self.cooldown_ok:
            return "missile_cooldown"
        if not self.target_not_saturated:
            return "target_saturated"
        return None

    def as_dict(self) -> dict:
        """Legacy mapping shape consumed by ``get_fire_feasibility`` callers."""
        return {
            "radar_lock": self.radar_lock,
            "datalink_lock": self.datalink_lock,
            "target_in_fov": self.target_in_fov,
            "gimbal_ok": self.gimbal_ok,
            "radar_range_ok": self.radar_range_ok,
            "weapon_range_ok": self.weapon_range_ok,
            "launch_range_ok": self.launch_range_ok,
            "inventory_ok": self.inventory_ok,
            "cooldown_ok": self.cooldown_ok,
            "target_not_saturated": self.target_not_saturated,
            "can_fire": self.can_fire,
            "veto_reason": self.veto_reason,
            "remaining_missiles": self.remaining_missiles,
        }


NO_TARGET_GATES = FireGates(error="no_target")


def _target_id(target) -> object | None:
    if isinstance(target, TacticalContact):
        return target.track_id
    return getattr(target, "id", None)


def _weapon_range_ok(aircraft, target, slant_range_m: float) -> bool:
    """Is the target inside the MISSILE's launch range, per the live DLZ?

    ``r_pi_m`` is the ceiling: beyond it the shot sits in DLZ zone R4 (too far, low
    Pk), which is the same boundary ``_shot_in_envelope`` scores a launch against.
    Using the live DLZ rather than a static per-type maximum means the gate follows
    closure, altitude and aspect, so standing off cold reads as out of range even at
    a distance a hot shot would reach.

    Cached per (target, step) on the aircraft: ``evaluate_fire_gates`` runs every
    step for every agent and previously did no DLZ work at all, so an uncached
    computation would put a WEZ solve in the hot loop.

    Unscorable envelopes return True (permissive). Refusing the shot instead would
    turn every DLZ hiccup into an invisible no-fire, which is the failure shape this
    module exists to prevent -- an unscorable envelope is already logged loudly on
    the launch path.
    """
    wez = getattr(aircraft, "wez", None)
    if wez is None or target is None:
        return True

    cache_key = (id(target), getattr(target, "track_id", None) or getattr(target, "id", None))
    cache = getattr(aircraft, "_weapon_range_cache", None)
    if cache is not None and cache[0] == cache_key and cache[1] == slant_range_m:
        return cache[2]

    try:
        if isinstance(target, TacticalContact):
            dlz = wez.compute_dlz_from_track(target.state, target.covariance).nominal
        else:
            dlz = wez.compute_dlz(target)
        r_pi_m = float(getattr(dlz, "r_pi_m", 0.0) or 0.0)
        result = slant_range_m <= r_pi_m if r_pi_m > 0.0 else True
    except Exception:
        result = True

    try:
        aircraft._weapon_range_cache = (cache_key, slant_range_m, result)
    except Exception:
        pass
    return result


def own_radar_lock_ok(aircraft, target) -> bool:
    """Does this aircraft hold its OWN radar lock on ``target`` right now?

    This is what ``tactical/lock_rate`` reports. It is a strict subset of the locks a
    launch will accept -- see the module docstring.
    """
    if target is None:
        return False
    sensor = getattr(aircraft, "sensor", None)
    if sensor is None:
        return False
    try:
        if isinstance(target, TacticalContact):
            return target.track_id in set(sensor.get_locked_targets() or ())
        has_radar_lock = getattr(sensor, "has_radar_lock", None)
        return bool(has_radar_lock(target)) if callable(has_radar_lock) else False
    except Exception:
        return False


def datalink_lock_ok(aircraft, target, simulator) -> bool:
    """Is a flight member (or AWACS) holding a lock this aircraft can shoot off?"""
    if target is None or simulator is None:
        return False
    try:
        from bvr_marl_core.radar.core.data_link import DataLink

        return _target_id(target) in DataLink.group_locked_target_ids(simulator, aircraft)
    except Exception:
        return False


def weapon_fov_ok(aircraft, target) -> bool:
    """Is ``target`` inside the weapon field of view right now?"""
    if target is None:
        return False
    weapons = getattr(aircraft, "weapons", None)
    if weapons is None:
        return False
    try:
        if isinstance(target, TacticalContact):
            return bool(weapons.is_contact_in_fov(target))
        in_fov = getattr(weapons, "is_target_in_fov", None)
        return bool(in_fov(target)) if callable(in_fov) else False
    except Exception:
        return False


def missile_cooldown_ok(aircraft) -> bool:
    """Has the post-launch cooldown expired?

    The authoritative timer lives in the action-space state dict, which the aircraft
    cannot see, so ``WeaponFiringHandler`` mirrors it onto the unit via
    :func:`sync_missile_cooldown` on every step. Absent the mirror this reports ready,
    which is the pre-existing behaviour of the observation path.
    """
    return float(getattr(aircraft, "missile_cooldown_left_s", 0.0) or 0.0) <= 0.0


def sync_missile_cooldown(aircraft, cooldown_left_s: float) -> None:
    """Publish the action-space cooldown timer where the observation path can read it.

    Without this the observation's ``can_fire`` and the launch gate disagree for the
    ``missile_cooldown_s`` seconds after every shot -- the window in which a policy that
    has just fired is most likely to press again.
    """
    try:
        aircraft.missile_cooldown_left_s = float(cooldown_left_s)
    except Exception:
        pass


def target_saturated(aircraft, target, simulator, max_missiles_per_target: int) -> bool:
    """Are there already ``max_missiles_per_target`` friendly missiles on this target?"""
    tid = _target_id(target)
    if tid is None or simulator is None:
        return False
    try:
        count = simulator.count_missiles_at_target(getattr(aircraft, "group", None), tid)
        return count >= max_missiles_per_target
    except Exception:
        return False


def evaluate_fire_gates(
    aircraft,
    target,
    *,
    simulator=None,
    obs_helper=None,
    max_missiles_per_target: int = 2,
) -> FireGates:
    """Evaluate every launch gate for ``aircraft`` against ``target``.

    Args:
        aircraft: The shooting unit.
        target: A ``TacticalContact`` or a raw unit. ``None`` yields
            :data:`NO_TARGET_GATES`.
        simulator: Needed for the datalink lock and the saturation count. Omitting it
            makes both gates report their permissive value, matching how the
            observation path behaved before those gates existed.
        obs_helper: Optional ``ObservationHelper`` used for the geometry (ATA / slant
            range). Falls back to ``aircraft.observation_helper`` and, failing that,
            leaves the geometric gates open rather than closing a shot the launch path
            would have taken.
        max_missiles_per_target: Per-target saturation cap supplied by the caller.

    Returns:
        A :class:`FireGates` whose ``can_fire`` is the one predicate both the
        observation bit and the shot-opportunity counter use.
    """
    if target is None:
        return NO_TARGET_GATES

    try:
        weapons = getattr(aircraft, "weapons", None)
        remaining = getattr(aircraft, "remaining_missiles", None)
        if not remaining:
            remaining = getattr(weapons, "remaining_missiles", 0) if weapons else 0
        remaining = int(remaining or 0)

        radar_lock = own_radar_lock_ok(aircraft, target)
        # Only worth the DataLink lookup when the aircraft has no lock of its own.
        datalink = False if radar_lock else datalink_lock_ok(aircraft, target, simulator)

        gimbal_ok = True
        radar_range_ok = True
        weapon_range_ok = True
        helper = obs_helper if obs_helper is not None else _helper_of(aircraft)
        if helper is not None:
            geom = helper.get_geometry_kinematics(target)
            if geom.get("valid"):
                radar = getattr(aircraft, "radar", None)
                h_fov = float(getattr(radar, "h_fov_deg", 0.0) or 0.0)
                max_range = float(getattr(radar, "max_range_m", 0.0) or 0.0)
                gimbal_ok = geom["ata_deg"] < (h_fov / 2.0) if h_fov else True
                radar_range_ok = (
                    geom["slant_range_m"] <= max_range * RADAR_RANGE_USABLE_FRACTION
                    if max_range
                    else True
                )
                weapon_range_ok = _weapon_range_ok(aircraft, target, geom["slant_range_m"])
            else:
                gimbal_ok = False
                radar_range_ok = False

        return FireGates(
            inventory_ok=remaining > 0,
            radar_lock=radar_lock,
            datalink_lock=datalink,
            target_in_fov=weapon_fov_ok(aircraft, target),
            gimbal_ok=gimbal_ok,
            radar_range_ok=radar_range_ok,
            weapon_range_ok=weapon_range_ok,
            cooldown_ok=missile_cooldown_ok(aircraft),
            target_not_saturated=not target_saturated(
                aircraft, target, simulator, max_missiles_per_target
            ),
            roe_ok=roe_engageable(target, aircraft=aircraft, simulator=simulator),
            remaining_missiles=remaining,
        )
    except Exception as e:  # pragma: no cover - defensive, matches prior behaviour
        return FireGates(error=f"error:{e}")


def roe_engageable(target, *, aircraft=None, simulator=None) -> bool:
    """False when the scenario forbids engaging this target (AWACS and other support).

    ONE definition, used by both the launch path and the feasibility gate. They used to
    disagree: `AircraftWeaponSystem.fire_missile_at_contact` refused a protected contact
    with `contact_non_engageable_roe`, while `evaluate_fire_gates` did not model the rule
    at all. So `can_fire` -- and with it the observation's can-fire bit, the
    shot-opportunity counter and the fire-gradient mask -- reported a shot that the weapon
    then silently refused. Measured on 1v1 self-play with the trigger held: 92 of 92
    gate-passing presses were rejected as `contact_non_engageable_roe`, for zero launches.

    Handles both target kinds:
      * a raw unit carries `is_non_engageable` directly;
      * a ``TacticalContact`` carries only a track id, so it is resolved through the
        evaluator boundary -- the same route the weapon uses. Nothing about the resolved
        unit reaches the weapon, its guidance, or the observation; only this boolean.

    Permissive on failure: with no simulator or no resolver the rule cannot be evaluated,
    and closing the gate there would forbid every shot rather than only protected ones.
    """
    if target is None:
        return False
    if getattr(target, "is_non_engageable", False):
        return False

    track_id = getattr(target, "track_id", None)
    if track_id is None or simulator is None:
        return True
    resolve = getattr(simulator, "evaluator_truth_id_for_contact", None)
    if not callable(resolve):
        return True
    try:
        truth_id = resolve(getattr(aircraft, "id", None), track_id)
    except Exception:
        return True
    if truth_id is None:
        return True
    unit = getattr(simulator, "active_units", {}).get(truth_id)
    return not bool(getattr(unit, "is_non_engageable", False))


def _helper_of(aircraft):
    for attr in ("observation_helper", "obs_helper"):
        helper = getattr(aircraft, attr, None)
        if helper is not None and hasattr(helper, "get_geometry_kinematics"):
            return helper
    return None
