from dataclasses import asdict

import numpy as np

from bvr_marl_core.domain.information import WeaponTrack
from bvr_marl_core.domain.sensing_visibility import is_sensor_invisible_to
from bvr_marl_core.missiles.core.engine import MissileEngine
from bvr_marl_core.missiles.core.movement import MissileMovement
from bvr_marl_core.missiles.core.phases import MissilePhaseManager
from bvr_marl_core.missiles.guidance.guidance import MissileGuidance
from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider
from bvr_marl_core.physics.missiles import MissilePhysics
from bvr_marl_core.radar.core.data_link import DataLink
from bvr_marl_core.radar.obs.observation import DEFAULT_NOTCH_VELOCITY_MPS
from bvr_marl_core.radar.units.missile import MissileRadar
from bvr_marl_core.simulator.core.events import Event
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.core.units import FlyingUnit
from bvr_marl_core.simulator.simulator import Simulator
from bvr_marl_core.simulator.utils.tick_path import pose_at_fraction, unit_tick_path

# Default warhead lethal-radius scale (m) for the range-dependent kill
# probability, applied to every missile so lethality is consistent across the
# arsenal. Pk(d) = P_fuze * exp(-(d / lethal_radius_m)^2) for a neutral
# target and track. A clean geometric intercept is limited by fuze reliability,
# while an opened terminal miss reduces Pk. Per-missile configs may override.
#
# BALANCE KNOB. At this scale a 250 m intercept retains about 37% of clean-hit
# lethality and a detonation at the 500 m outer fuze boundary about 1.8%.
#
# The previous note quoted 78% and 37% for those two distances. Those are the
# values for a 500 m scale, not this one: the constant was halved to decouple
# lethality from the fuze radius and the note was never updated, so it overstated
# lethality by roughly 2x at 250 m and 20x at the fuze boundary.
#
# The "~200-400 m closest approach" the old note sized against was itself a
# symptom rather than a property of the sim. The weapon's own seeker acquires in
# the endgame but numbered its tracks in a different namespace than the contact it
# was cued against, so `track_only` discarded them and the missile dead-reckoned
# its last several seconds. With endgame re-association (see
# TargetProvider._reassociate_seeker_tracks) measured closest approach is 0.5-6 m.
DEFAULT_LETHAL_RADIUS_M = 250.0

# Default arming delay (s) after launch before the proximity fuze is live, so a
# reliable close-range CCD cannot score a kill the instant a missile is fired.
DEFAULT_ARMING_TIME_S = 1.5

# Neutral defaults for the weapon-effectiveness submodel parameters. A missile
# that does not set these behaves exactly as the legacy scalar kill model
# (Pk = hit_probability * exp(-(d/lethal_radius)^2)); per-missile configs
# override them with physically meaningful values. See
# bvr_marl_core.simulator.core.effectiveness for how each term enters Pk.
DEFAULT_FUZE_RELIABILITY = 1.0
DEFAULT_GUIDANCE_RELIABILITY = 1.0
DEFAULT_SEEKER_RELIABILITY = 1.0
DEFAULT_DATALINK_RELIABILITY = 1.0


class Missile(FlyingUnit):
    unit_kind: str = "missile"

    def __init__(
        self,
        name: str,
        firing_time_s,
        target,
        source,
        map_limits,
        group: str,
        config: dict,
        data_link_mode: str = "full",
    ):
        cfg = asdict(config) if not isinstance(config, dict) else config

        super().__init__(
            name=name,
            position=source.position.copy(),
            yaw_deg=source.yaw_deg,
            speed=source.speed,
            pitch_deg=source.pitch_deg,
            roll_deg=0.0,
        )

        self.type = "Missile"
        self.source = source
        self.weapon_track = target if isinstance(target, WeaponTrack) else None
        # Legacy/oracle constructors may still pass a Unit. The operational
        # sensor-limited launch path passes only WeaponTrack and stores no Unit.
        self.target = None if self.weapon_track is not None else target
        # The contact this weapon was committed to at launch. Immutable for the whole
        # flight, unlike designated_track_id, which follows a track the seeker's
        # tracker may retire and re-issue. The launch path overwrites these with the
        # firing platform's identity; a weapon built directly falls back to its own
        # weapon track.
        self.launch_sensor_id = None
        self.launch_contact_id = None
        self.launch_report_lineage = ()
        if self.weapon_track is not None:
            self.launch_contact_id = self.weapon_track.snapshot.track_id
            self.launch_report_lineage = self.weapon_track.snapshot.report_lineage
            self.designated_target_id = self.weapon_track.snapshot.track_id
            self.designated_track_id = self.weapon_track.snapshot.track_id
        self.group = group if group else source.group
        self.map_limits = map_limits
        self.firing_time_s = firing_time_s
        self.max_speed_mps = cfg["max_speed_mps"]
        # Kinematic max range (cited, head-on/high-altitude) — the DLZ anchor.
        # Distinct from the radar/seeker activation range (cfg["radar"]["max_range_m"]).
        self.max_range_m = cfg.get("max_range_m", cfg["radar"]["max_range_m"])
        self.hit_probability = cfg["hit_probability"]
        # Warhead lethal-radius scale for the range-dependent kill probability
        # (consistent arsenal-wide default; per-missile config may override).
        self.lethal_radius_m = cfg.get("lethal_radius_m", DEFAULT_LETHAL_RADIUS_M)
        # Arming delay: the fuze is inert until this many seconds after launch.
        self.arming_time_s = cfg.get("arming_time_s", DEFAULT_ARMING_TIME_S)

        # --- Weapon-effectiveness submodel parameters -----------------------
        # These decompose terminal lethality into physically meaningful terms.
        # Guidance is resolved geometrically by the authoritative CPA path and
        # remains telemetry rather than being multiplied a second time:
        #   P_intercept  = geometric CPA/fuze gate (already achieved here)
        #   P_detonation ~ fuze_reliability (proximity fuze functions)
        #   P_damage     ~ warhead_effectiveness (lethal effect at miss = 0)
        #   P_kill       = the product (see KillProbabilityModel)
        # seeker/datalink reliability are exposed for the terminal-track model
        # (Phase 7) and observations; defaults are neutral (no behaviour change).
        self.warhead_effectiveness = cfg.get("warhead_effectiveness", 1.0)
        self.fuze_reliability = cfg.get("fuze_reliability", DEFAULT_FUZE_RELIABILITY)
        self.guidance_reliability = cfg.get("guidance_reliability", DEFAULT_GUIDANCE_RELIABILITY)
        self.seeker_reliability = cfg.get("seeker_reliability", DEFAULT_SEEKER_RELIABILITY)
        self.datalink_reliability = cfg.get("datalink_reliability", DEFAULT_DATALINK_RELIABILITY)
        # Normalised terminal track uncertainty (C_trk in [0, 1]) consumed by the
        # P_trk kill submodel. None == unknown/neutral (no covariance penalty). The
        # guidance target provider populates this live from the seeker track's
        # position covariance, so a noisy/jammed estimate lowers terminal Pk.
        self.terminal_track_uncertainty: float | None = None
        # Next-gen missiles keep a full two-way datalink for the whole flight, so
        # they inherit the launching side's cooperative triangulation on a jammer
        # even in the terminal phase. Classic Fox-3s (default) go seeker-autonomous
        # terminal -> home-on-jam when the target is jamming.
        self.full_datalink = bool(cfg.get("full_datalink", False))
        self.life_time_s = cfg["life_time_s"]
        self.motor_burn_s = cfg.get("motor_burn_s", 30.0)
        self.elapsed_time_s = 0.0
        self.is_missile = True
        # Deferred removal: flagged here, swept by the simulator AFTER hit
        # detection so a same-tick proximity hit is never lost (see update()).
        self.should_be_removed = False
        self.removal_reason: str | None = None
        # Swept (lat, lon, alt) positions over the current tick's sub-stepped
        # flight; consumed by the hit detector to walk the missile's real path.
        self.tick_path: list[tuple[float, float, float]] = []

        self.desired_yaw_deg = self.yaw_deg
        self.desired_pitch_deg = self.pitch_deg
        self.latest_target_velocity = np.zeros(3)

        # Propellant mass burns off over the motor burn (see MissileEngine). Launch
        # mass is the configured mass_kg; burnout mass is what remains once the
        # propellant is spent. Default propellant fraction ~35% of launch mass; a
        # type can override propellant_mass_kg for a specific motor.
        self.launch_mass_kg = float(cfg["mass_kg"])
        self.propellant_mass_kg = float(cfg.get("propellant_mass_kg", 0.35 * cfg["mass_kg"]))
        self.burnout_mass_kg = max(
            self.launch_mass_kg - self.propellant_mass_kg, 0.4 * self.launch_mass_kg
        )

        # Countermeasure seduction: set to the countermeasure object that has
        # decoyed this missile (see missiles/countermeasures.py). Once set, guidance
        # homes on it instead of the aircraft.
        self.seduced_by = None
        # Sensor-limited missiles must not store the countermeasure Unit either.
        # The physical seduction resolver freezes only its observed position.
        self.seduced_position = None

        # Guidance law: "pn" (proportional navigation, default) or "apn" (augmented
        # PN, which adds a target-acceleration term to lead a maneuvering target).
        # Read by PnPropNavGuidance at construction, so it must be set beforehand.
        self.use_apn = bool(cfg.get("use_apn", str(cfg.get("guidance_law", "pn")).lower() == "apn"))

        self.phase_manager = MissilePhaseManager(
            cfg.get("flight_phases"),
            life_time_s=cfg["life_time_s"],
            motor_burn_s=cfg.get("motor_burn_s", 30.0),
        )
        self.engine = MissileEngine(self, cfg.get("motor_burn_s", 30.0))
        self.physics = self._init_physics(cfg)
        self.movement = MissileMovement(self, self.physics)
        self.radar = self._init_radar(cfg["radar"], data_link_mode)
        self.target_provider = GuidanceTargetProvider(self)
        self.radar.target_provider = self.target_provider
        self.guidance = MissileGuidance(self, self.target_provider)
        self.oracle_direct_launch = False

    def update(self, tick_secs: float, sim: Simulator) -> list[Event]:
        self.elapsed_time_s += tick_secs
        self.phase_manager.update(self.elapsed_time_s)
        self.engine.update(tick_secs, sim)

        # Countermeasure seduction: a chaff/decoy (or, for IR seekers, a flare) may
        # pull this missile off the target and onto the false return.
        from bvr_marl_core.missiles.countermeasures import evaluate_seduction

        evaluate_seduction(self, sim, tick_secs)

        radar_targets = []
        for unit in sim.active_units.values():
            if (
                hasattr(unit, "group")
                and unit.group != self.group
                and not getattr(unit, "is_missile", False)
                and not getattr(unit, "is_countermeasure", False)
                and not getattr(unit, "is_non_engageable", False)
                # A seeker cannot acquire what no sensor can see. `is_non_engageable`
                # above is the ROE rule and is not a substitute: a scenario may set one
                # without the other, and only this flag is about being SEEN.
                and not is_sensor_invisible_to(unit, self.group)
            ):
                radar_targets.append(unit)

        operational_track_only = self.weapon_track is not None
        substep_sensor_target = None
        substep_track_target = (
            None if operational_track_only else self._resolve_substep_guidance_target(sim)
        )
        if substep_track_target is None:
            self.radar.update(
                tick_secs,
                sim,
                targets=radar_targets,
                owner_position=self.position,
            )
            # The seeker sensor may sample the detected target's staged path at
            # flight substeps. The Unit is ephemeral sensor-simulation input; it
            # is never stored by guidance or copied into the WeaponTrack.
            substep_sensor_target = self._resolve_substep_sensor_target(
                sim, radar_targets, operational_track_only
            )
        else:
            try:
                self.radar._manage_datalink_mode()
                self.radar.param_policy.update_dynamic_params()
                # Option A: while the seeker has a stable own track, guide on it
                # alone; once it has lost the target, stage a fused datalink
                # (fighter/AWACS) measurement so a coasting substep can re-acquire
                # from the other sources — "lost track -> check all sources".
                if self.radar.has_stable_seeker_track():
                    self.radar._pending_datalink_meas = None
                    self.radar._datalink_meas_used = False
                else:
                    self.radar.prepare_datalink_reacquire(
                        sim, self.position, getattr(substep_track_target, "id", None)
                    )
            except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
                pass

        _ti = self.radar.get_tracker_info()
        if _ti is not None and _ti.get("velocity") is not None:
            try:
                self.latest_target_velocity = np.array(_ti["velocity"], dtype=float)
            except (TypeError, ValueError):
                pass

        # Authoritative sub-stepped flight: integrate seeker-track refresh,
        # guidance, and movement in fine steps. A single 1 s guidance step holds
        # one heading across a ~1.4 km endgame and passes hundreds of metres wide;
        # sub-stepping lets the missile continuously re-aim so clean shots
        # actually intercept. self.tick_path records the swept positions for the
        # hit check, which walks the missile's *real* path rather than a chord.
        n_sub = self._flight_substeps()
        sub_dt = tick_secs / n_sub
        self.tick_path = [self._pose()]
        self._last_cpa_event = None
        if hasattr(self.radar, "begin_designated_tick"):
            self.radar.begin_designated_tick()
        for i in range(n_sub):
            substep_target = (
                substep_sensor_target
                if operational_track_only
                else substep_track_target or self._resolve_substep_guidance_target(sim)
            )
            target_position = self._target_pose_at_fraction(
                sim,
                substep_target,
                i / n_sub,
            )
            self.substep_update(
                sub_dt,
                sim,
                target_position=target_position,
                sensor_target=substep_target,
            )
            self.tick_path.append(self._pose())

        # Defer removal: flag the missile but keep it in the sim for this tick so
        # the substepper can still register a same-tick proximity hit. The
        # simulator sweeps flagged units (emitting a removal event) AFTER hit
        # detection. Removing here directly would delete a missile that reaches
        # its target on the very tick it also runs out of energy/lifetime.
        reason = self._removal_reason()
        if reason is not None:
            self.should_be_removed = True
            self.removal_reason = reason

        return []

    @property
    def guidance_track_id(self):
        """The track this weapon is prosecuting *now*.

        Distinct from ``launch_contact_id``, which is fixed at commit time: the seeker's
        tracker may retire and re-issue the track while following the same aircraft, so
        this changes during a flight and the launch identity does not. Distinct again
        from the physical unit, which is evaluator-only and never stored on a weapon.
        """
        provider_id = getattr(self.target_provider, "current_target_id", None)
        if provider_id is not None:
            return provider_id
        return getattr(self, "designated_track_id", None)

    def _pose(self) -> tuple[float, float, float]:
        p = self.position
        return (float(p.lat), float(p.lon), float(p.alt))

    def _resolve_substep_sensor_target(
        self, sim, radar_targets, operational_track_only: bool
    ) -> object | None:
        """Resolve the physical unit whose staged path the seeker samples this tick.

        A WeaponTrack-guided missile holds an anonymous operational track ID. That
        namespace is disjoint from simulator unit IDs, so the track ID must never
        be used as a roster key: the two can collide numerically and put an
        unrelated -- or friendly -- unit into the terminal candidate set. Its
        physical target therefore comes only from the evaluator-side weapon-target
        association, and a missing association yields no candidate rather than a
        different unit.
        """
        if operational_track_only:
            resolve = getattr(sim, "evaluator_target_for_weapon", None)
            target = resolve(self) if callable(resolve) else None
            if target is None:
                record = getattr(sim, "record_diagnostic", None)
                departed = getattr(sim, "evaluator_weapon_target_departed", None)
                if callable(departed) and departed(self):
                    self.should_be_removed = True
                    self.removal_reason = "target_destroyed"
                    if callable(record):
                        record("weapon_target_departed")
                elif callable(record):
                    record("missile_terminal_association_missing")
                return None
            # Identity, never equality: Unit compares by value, and the roster
            # already excludes destroyed, friendly, and non-engageable units.
            return target if any(unit is target for unit in radar_targets) else None

        # Oracle direct launch only. Here locked_target genuinely holds a physical
        # unit ID, because no operational contact ever selected this weapon's target.
        locked_id = getattr(self.radar, "get_locked_target", lambda: None)()
        if locked_id is None and self.oracle_direct_launch:
            locked_id = self.designated_target_id
        if locked_id is None:
            return None
        return next(
            (target for target in radar_targets if getattr(target, "id", None) == locked_id),
            None,
        )

    def _resolve_substep_guidance_target(self, sim) -> object | None:
        active_units = getattr(sim, "active_units", {}) or {}

        def _lookup(unit_id):
            if unit_id is None:
                return None
            if hasattr(active_units, "get"):
                unit = active_units.get(unit_id)
                if unit is not None:
                    return unit
            values = active_units.values() if hasattr(active_units, "values") else active_units
            for unit in values:
                if getattr(unit, "id", None) == unit_id:
                    return unit
            return None

        candidate_ids = []
        try:
            if hasattr(self.radar, "get_locked_target"):
                candidate_ids.append(self.radar.get_locked_target())
        except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
            pass
        candidate_ids.append(getattr(self.target_provider, "current_target_id", None))
        candidate_ids.append(getattr(self, "designated_target_id", None))
        candidate_ids.append(getattr(getattr(self, "target", None), "id", None))

        for target_id in candidate_ids:
            unit = _lookup(target_id)
            if self._is_valid_substep_guidance_target(unit):
                return unit

        fallback = getattr(self, "target", None)
        if self._is_valid_substep_guidance_target(fallback):
            return fallback
        return None

    def _is_valid_substep_guidance_target(self, unit) -> bool:
        if unit is None or not hasattr(unit, "position"):
            return False
        if getattr(unit, "is_destroyed", False):
            return False
        if getattr(unit, "is_missile", False):
            return False
        if getattr(unit, "is_countermeasure", False):
            return False
        if getattr(unit, "is_non_engageable", False):
            return False
        unit_group = getattr(unit, "group", None)
        if unit_group is not None and unit_group == self.group:
            return False
        return True

    def _target_pose_at_fraction(self, sim, target, fraction: float) -> Position | None:
        if target is None or not hasattr(target, "position"):
            return None

        # Prefer the arc the target actually swept this tick. Interpolating
        # between its begin- and end-of-tick poses cuts the corner off every
        # turn, which biases guidance toward the inside of a defensive break.
        real_path = unit_tick_path(target, sim)
        if real_path is not None:
            lat, lon, alt = pose_at_fraction(real_path, fraction)
            return Position(lat, lon, alt)

        staged_position = getattr(sim, "staged_next_position", lambda _unit: None)(target)
        end = staged_position if staged_position is not None else target.position
        begin = end
        ccd = getattr(sim, "ccd", None)
        if ccd is not None and hasattr(ccd, "get_begin_state"):
            try:
                state = ccd.get_begin_state(target)
                if state is not None and state.get("pos", None) is not None:
                    begin = state["pos"]
            except Exception:
                begin = end

        f = max(0.0, min(1.0, float(fraction)))
        return Position(
            float(begin.lat) + (float(end.lat) - float(begin.lat)) * f,
            float(begin.lon) + (float(end.lon) - float(begin.lon)) * f,
            float(begin.alt) + (float(end.alt) - float(begin.alt)) * f,
        )

    # Adaptive flight sub-steps: few while cruising, many in the terminal endgame.
    _MIN_FLIGHT_SUBSTEPS = 2
    _MAX_FLIGHT_SUBSTEPS = 12
    _SUBSTEP_NEAR_M = 1500.0
    _SUBSTEP_FAR_M = 10000.0

    def _flight_substeps(self) -> int:
        """Number of guidance+movement sub-steps for this tick, ramped by range to
        the guidance target so terminal accuracy is high without paying for it in
        the long midcourse cruise."""
        import math as _math

        try:
            tp = self.target_provider.get_guidance_target()
            if tp is None:
                return self._MIN_FLIGHT_SUBSTEPS
            cos_lat = _math.cos(_math.radians(self.position.lat))
            de = (tp.lon - self.position.lon) * 111_000.0 * cos_lat
            dn = (tp.lat - self.position.lat) * 111_000.0
            du = tp.alt - self.position.alt
            rng = _math.sqrt(de * de + dn * dn + du * du)
        except Exception:
            return self._MIN_FLIGHT_SUBSTEPS
        near, far = self._SUBSTEP_NEAR_M, self._SUBSTEP_FAR_M
        lo, hi = self._MIN_FLIGHT_SUBSTEPS, self._MAX_FLIGHT_SUBSTEPS
        if rng <= near:
            return hi
        if rng >= far:
            return lo
        frac = (far - rng) / (far - near)  # 0 at far -> 1 at near
        return int(round(lo + frac * (hi - lo)))

    def _removal_reason(self) -> str | None:
        """
        Determine why the missile should be removed, or None to keep it alive.
        Evaluated after movement so the position/energy reflect this tick.
        """
        # Out of usable energy (speed/altitude floor) — checked first so a spent
        # missile is retired even while still nominally within its lifetime.
        if self.engine.should_remove_missile_on_energy(self):
            return "energy_depleted"

        if hasattr(self, "map_limits") and self.map_limits is not None:
            lat_violated = (
                self.position.lat <= self.map_limits.bottom_lat
                or self.position.lat >= self.map_limits.top_lat
            )
            lon_violated = (
                self.position.lon <= self.map_limits.left_lon
                or self.position.lon >= self.map_limits.right_lon
            )
            if lat_violated or lon_violated:
                return "boundary"

        # `life_time_s` is the single authority on flight time. There used to be a
        # second, hardcoded cap here -- retire the missile once it had glided for
        # more than 120 s past `motor_burn_s` -- which silently overrode the
        # per-type configuration for every weapon whose lifetime exceeds
        # `motor_burn_s + 120`: r37m died at 140 s against a configured 180,
        # k77m at 132 against 150, and default_missile at 152 against 220. It
        # bound on 3 of the 6 fox3 types and on none of the other 3, so the
        # constant was a leftover default rather than a modelled limit. Spent
        # weapons are already retired by the energy floor checked above.
        life_time_s = getattr(self, "life_time_s", 100.0)
        if self.elapsed_time_s >= life_time_s:
            return "lifetime_expired"

        # Only remove if lost lock for extended period with no fuel — not on transient lock loss.
        locked_target = None
        has_guidance = False

        if hasattr(self.radar, "get_locked_target"):
            try:
                locked_target = self.radar.get_locked_target()
            except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
                pass

        if hasattr(self, "target_provider"):
            try:
                guidance_pos = self.target_provider.get_guidance_target()
                has_guidance = guidance_pos is not None
            except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
                pass

        if (
            locked_target is None
            and not has_guidance
            and self.engine.fuel_s == 0.0
            and self.elapsed_time_s > 120.0
        ):
            return "lock_lost"

        return None

    def _init_physics(self, cfg: dict) -> MissilePhysics:
        return MissilePhysics(
            MissilePhysics.Params(
                mass_kg=cfg["mass_kg"],
                reference_area_m2=cfg["reference_area_m2"],
                aspect_ratio=cfg["aspect_ratio"],
                oswald_e=cfg["oswald_e"],
                cd0=cfg["drag_coefficient"],
                constant_engine_F=cfg["constant_engine_N"],
                n_max=cfg["n_max"],
                max_speed_mps=cfg["max_speed_mps"],
                drag_scale=cfg.get("drag_scale", 1.0),
            )
        )

    def _init_radar(self, rc: dict, data_link_mode: str) -> MissileRadar:
        return MissileRadar(
            horizontal_fov_deg=rc["horizontal_fov_deg"],
            vertical_fov_deg=rc["vertical_fov_deg"],
            max_range_m=rc["max_range_m"],
            radar_frequency_hz=rc["radar_frequency_hz"],
            tx_power_w=rc["tx_power_w"],
            antenna_gain_db=rc["antenna_gain_db"],
            snr_threshold_db=rc["snr_threshold_db"],
            false_alarm_rate=rc.get("false_alarm_rate", 0.005),
            range_resolution_m=rc.get("range_resolution_m", 100.0),
            angular_resolution_deg=rc.get("angular_resolution_deg", 2.0),
            lut_bins=rc.get("lut_bins", (256, 256)),
            owner=self,
            data_link=DataLink(data_link_mode),
            initial_datalink_mode="other",
            original_data_link_mode=data_link_mode,
            notch_velocity_mps=rc.get("notch_velocity_mps", DEFAULT_NOTCH_VELOCITY_MPS),
            processing_gain_db=rc.get("processing_gain_db", 33.0),
        )

    def substep_update(self, dt: float, sim, target_position=None, sensor_target=None) -> list:
        if target_position is not None and hasattr(self.radar, "update_designated_track_substep"):
            try:
                self.radar.update_designated_track_substep(
                    dt,
                    sensor_target,
                    self.position,
                    target_position=target_position,
                )
            except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
                pass

        try:
            self.target_provider.update(sim, dt)
        except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
            pass

        yaw, pch = self.guidance.compute_guidance(
            self,
            self.target_provider,
            getattr(self.radar, "tracker_manager", None),
            dt,
        )
        self.desired_yaw_deg = yaw
        self.desired_pitch_deg = pch

        self.movement.update(dt)
        return []
