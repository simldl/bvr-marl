from dataclasses import asdict

import numpy as np

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

# Default warhead lethal-radius scale (m) for the range-dependent kill
# probability, applied to every missile so lethality is consistent across the
# arsenal. Pk(d) = hit_probability * exp(-(d / lethal_radius_m)^2); a clean
# intercept (miss ~ 0) keeps the flat hit_probability, while an opened terminal
# miss (~150 m) drops Pk sharply. Per-missile configs may override.
DEFAULT_LETHAL_RADIUS_M = 100.0

# Default arming delay (s) after launch before the proximity fuze is live, so a
# reliable close-range CCD cannot score a kill the instant a missile is fired.
DEFAULT_ARMING_TIME_S = 1.5


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
        self.target = target
        self.group = group if group else source.group
        self.map_limits = map_limits
        self.firing_time_s = firing_time_s
        self.max_speed_mps = cfg["max_speed_mps"]
        self.hit_probability = cfg["hit_probability"]
        # Warhead lethal-radius scale for the range-dependent kill probability
        # (consistent arsenal-wide default; per-missile config may override).
        self.lethal_radius_m = cfg.get("lethal_radius_m", DEFAULT_LETHAL_RADIUS_M)
        # Arming delay: the fuze is inert until this many seconds after launch.
        self.arming_time_s = cfg.get("arming_time_s", DEFAULT_ARMING_TIME_S)
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

    def update(self, tick_secs: float, sim: Simulator) -> list[Event]:
        self.elapsed_time_s += tick_secs
        self.phase_manager.update(self.elapsed_time_s)
        self.engine.update(tick_secs, sim)

        radar_targets = []
        for unit in sim.active_units.values():
            if (
                hasattr(unit, "group")
                and unit.group != self.group
                and not getattr(unit, "is_missile", False)
                and not getattr(unit, "is_countermeasure", False)
                and not getattr(unit, "is_non_engageable", False)
            ):
                radar_targets.append(unit)

        substep_track_target = self._resolve_substep_guidance_target(sim)
        if substep_track_target is None:
            self.radar.update(
                tick_secs,
                sim,
                targets=radar_targets,
                owner_position=self.position,
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
            except Exception:
                pass

        _ti = self.radar.get_tracker_info()
        if _ti is not None and _ti.get("velocity") is not None:
            try:
                self.latest_target_velocity = np.array(_ti["velocity"], dtype=float)
            except Exception:
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
            substep_target = substep_track_target or self._resolve_substep_guidance_target(sim)
            target_position = self._target_pose_at_fraction(
                sim,
                substep_target,
                i / n_sub,
            )
            self.substep_update(sub_dt, sim, target_position=target_position)
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

    def _pose(self) -> tuple[float, float, float]:
        p = self.position
        return (float(p.lat), float(p.lon), float(p.alt))

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
        except Exception:
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

        end = target.position
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

        life_time_s = getattr(self, "life_time_s", 100.0)
        if self.elapsed_time_s >= life_time_s:
            return "lifetime_expired"

        if self.engine.fuel_s == 0.0:
            motor_burn_s = getattr(self, "motor_burn_s", 12.0)
            glide_time = self.elapsed_time_s - motor_burn_s
            max_glide_time = 120.0

            if glide_time > max_glide_time:
                return "glide_timeout"

        # Only remove if lost lock for extended period with no fuel — not on transient lock loss.
        locked_target = None
        has_guidance = False

        if hasattr(self.radar, "get_locked_target"):
            try:
                locked_target = self.radar.get_locked_target()
            except Exception:
                pass

        if hasattr(self, "target_provider"):
            try:
                guidance_pos = self.target_provider.get_guidance_target()
                has_guidance = guidance_pos is not None
            except Exception:
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
        )

    def substep_update(self, dt: float, sim, target_position=None) -> list:
        if target_position is not None and hasattr(self.radar, "update_designated_track_substep"):
            try:
                target = self._resolve_substep_guidance_target(sim)
                self.radar.update_designated_track_substep(
                    dt,
                    target,
                    self.position,
                    target_position=target_position,
                )
            except Exception:
                pass

        try:
            self.target_provider.update(sim, dt)
        except Exception:
            pass

        yaw, pch = self.guidance.compute_guidance(
            self, self.target_provider, self.radar.tracker_manager, dt
        )
        self.desired_yaw_deg = yaw
        self.desired_pitch_deg = pch

        self.movement.update(dt)
        return []
