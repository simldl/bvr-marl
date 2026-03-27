"""
Simplified action processor for the lightweight training environment.

Action vector (4 dims, all in [0, 1]):
  [0] energy_cmd   — throttle / specific excess power command
  [1] lift_phi     — lift-vector roll angle command
  [2] lift_N       — load-factor command
  [3] missile_trig — missile trigger (>= threshold fires)

Target selection is automatic (nearest alive enemy). No gun, no countermeasures.
Missiles are fired via fire_missile_direct(), which bypasses radar-lock gates.
"""

import logging
import math

import numpy as np

from air_to_air_rl.rl.environment.spaces.action_space.automation.weapon_cooldowns import (
    WeaponCooldowns,
)
from air_to_air_rl.rl.environment.spaces.action_space.base_processor import ActionProcessorBase
from air_to_air_rl.rl.environment.spaces.action_space.utils.target_sorting import TargetSorter

logger = logging.getLogger(__name__)

# Simplified action-space shape
SIMPLIFIED_ACTION_DIM = 4


class SimplifiedActionProcessor(ActionProcessorBase):
    """
    Lightweight action processor with a 4-dimensional action space.

    Differences from ActionProcessor:
      - No target-selection action; target chosen automatically (closest enemy).
      - No gun or countermeasure actions.
      - Fires missiles via fire_missile_direct() (no radar lock required).
      - Action indices: [0]=energy, [1]=lift_phi, [2]=lift_N, [3]=missile_trigger
    """

    def __init__(self, simulator):
        super().__init__(simulator)
        # Disable gun (not used in simplified env)
        self.enable_gun = False

    # Public API

    def apply(self, agent_id: str, action: np.ndarray):
        """Apply 4-dim action to the aircraft."""
        unit = self.simulator.active_units[agent_id]
        self._init_agent_state(agent_id, unit)
        state = self.agent_states[agent_id]

        self.missile_auto.cleanup_destroyed_targets(set(self.simulator.active_units.keys()))
        self.debug_collector.reset_step_counters(state)

        dt = float(getattr(self.simulator, "tick_secs", 1.0))
        action = np.clip(action, 0.0, 1.0)

        # --- Flight control ---
        u_tilde, u_hat = self._apply_action_filtering_simplified(action, state, dt, unit)

        s_factor, c_factor = self.envelope_calc.compute_envelope_scalars(
            unit, state["v_bar"], unit.position.alt
        )

        Ps_cmd = self.energy_proc.process_energy_command(
            unit, action[0], state, dt, s_factor, c_factor
        )
        new_throttle = self.energy_proc.compute_throttle_from_ps(
            unit, Ps_cmd, state, state["n_cmd_filtered"]
        )

        alpha_T = self._get_alpha_dt_aware(dt, self.tau_T)
        state["throttle_filtered"] = (1 - alpha_T) * state[
            "throttle_filtered"
        ] + alpha_T * new_throttle
        unit.control.set_throttle(state["throttle_filtered"])

        self.lift_vector_proc.process_lift_vector_commands(
            unit, action[1], action[2], state, dt, s_factor, c_factor
        )

        # --- Automatic target selection (closest enemy) ---
        selected_target = self._auto_select_target(unit)

        # --- Missile firing ---
        self.weapon_cooldowns.update_cooldowns(state, dt)
        self._handle_missile_firing(unit, action[3], selected_target, state)

        # Store visualization state
        self.state[unit.id] = {
            "ps_cmd_filtered": Ps_cmd,
            "n_cmd_filtered": state["n_cmd_filtered"],
            "phi_cmd_filtered": state["phi_cmd_filtered"],
            "throttle_filtered": state["throttle_filtered"],
        }

    # Internal helpers

    def _apply_action_filtering_simplified(self, action: np.ndarray, state: dict, dt: float, unit):
        """Deadzone + exponential filter for the 4-dim simplified action."""
        u_tilde = 2.0 * action - 1.0
        u_hat = u_tilde.copy()
        u_hat[1] = self.deadzone_filter.apply_deadzone(u_tilde[1], 1)
        u_hat[2] = self.deadzone_filter.apply_deadzone(u_tilde[2], 2)

        alpha_v = self._get_alpha_dt_aware(dt, self.tau_v)
        alpha_ang = self._get_alpha_dt_aware(dt, self.lift_vector_proc.tau_ang)

        # u_bar is kept at 10 dims for base-class compatibility; only first 3 matter here
        alphas = np.array([alpha_v, alpha_ang, alpha_ang] + [0.0] * 7)
        state["u_bar"] = (1 - alphas) * state["u_bar"] + alphas * np.concatenate(
            [u_hat[:3], np.zeros(7)]
        )
        state["v_bar"] = (1 - alpha_v) * state["v_bar"] + alpha_v * unit.speed

        return u_tilde, u_hat

    def _auto_select_target(self, unit):
        """Return the most dangerous non-saturated alive enemy (or None).

        Danger is ranked by a weighted combination of aspect angle (head-on =
        most dangerous), speed, altitude, and proximity.  Targets that are
        already saturated with friendly inbound missiles are excluded entirely.
        If every candidate is saturated, return None and do not fire.
        """
        candidates = [
            u
            for u in self.simulator.active_units.values()
            if u.id != unit.id
            and u.group != unit.group
            and not getattr(u, "is_missile", False)
            and not getattr(u, "is_non_engageable", False)
        ]
        if not candidates:
            return None

        max_inbound = getattr(self.weapon_cooldowns, "max_missiles_per_target", 2)

        # Exclude targets already saturated with friendly inbound missiles.
        unsaturated = [
            c
            for c in candidates
            if self.simulator.count_missiles_at_target(unit.group, c.id) < max_inbound
        ]
        if not unsaturated:
            return None

        # Sort by descending danger score; use id as a stable tiebreaker.
        unsaturated.sort(key=lambda c: (-self._danger_score(c, unit), c.id))
        return unsaturated[0]

    @staticmethod
    def _danger_score(enemy, own) -> float:
        """Danger posed by *enemy* to *own* — higher is more threatening.

        Components (each normalised to O(1)):
          aspect_factor  — 1.0 = head-on, 0.0 = tail-on
          speed_norm     — faster enemy carries more kinetic energy
          alt_norm       — higher enemy holds potential energy / positional advantage

        The combined numerator is divided by normalised distance so that closer
        threats score higher.  Speed and altitude are weighted slightly above
        aspect angle because they represent sustained capability rather than a
        transient geometry snapshot.
        """
        _SPEED_REF = 350.0  # m/s  (roughly Mach 1 at altitude)
        _ALT_REF = 12_000.0  # m    (typical combat ceiling)
        _DIST_REF = 50_000.0  # m    (50 km normalisation reference)

        # LOS vector from enemy to own aircraft, in metres (East / North / Up).
        dlat_m = (own.position.lat - enemy.position.lat) * 111_000.0
        dlon_m = (own.position.lon - enemy.position.lon) * 111_000.0
        dalt_m = own.position.alt - enemy.position.alt
        dist_m = math.sqrt(dlat_m**2 + dlon_m**2 + dalt_m**2)

        # Aspect angle factor: dot product of enemy velocity unit vector with
        # the LOS unit vector (enemy → own).  Dot = +1 means the enemy is
        # flying directly toward us (head-on) — most dangerous.
        if dist_m > 0.0 and enemy.speed > 0.0:
            los = (dlat_m / dist_m, dlon_m / dist_m, dalt_m / dist_m)
            ev = enemy.velocity  # namedtuple Velocity(vx=East, vy=North, vz=Up)
            ev_unit = (ev.vx / enemy.speed, ev.vy / enemy.speed, ev.vz / enemy.speed)
            dot = ev_unit[0] * los[0] + ev_unit[1] * los[1] + ev_unit[2] * los[2]
            aspect_factor = (dot + 1.0) / 2.0  # map [-1, 1] → [0, 1]
        else:
            aspect_factor = 1.0  # co-located or stationary → worst-case

        speed_norm = enemy.speed / _SPEED_REF
        alt_norm = max(0.0, enemy.position.alt) / _ALT_REF

        numerator = 1.0 * aspect_factor + 1.5 * speed_norm + 1.5 * alt_norm
        dist_norm = max(dist_m, 100.0) / _DIST_REF  # floor avoids division by zero
        return numerator / dist_norm

    @staticmethod
    def _range_sq(unit_a, unit_b) -> float:
        dlat = (unit_a.position.lat - unit_b.position.lat) * 111_000.0
        dlon = (unit_a.position.lon - unit_b.position.lon) * 111_000.0
        dalt = unit_a.position.alt - unit_b.position.alt
        return dlat**2 + dlon**2 + dalt**2

    # Fire-control FOV limits (half-angles): 120° × 30° cone in front of the nose.
    _FIRE_FOV_AZ_DEG: float = 60.0  # ± from heading  → 120° total horizontal
    _FIRE_FOV_EL_DEG: float = 15.0  # ± from pitch    →  30° total vertical

    @staticmethod
    def _is_in_fov(unit, target) -> bool:
        """Return True if *target* lies inside the shooter's fire-control FOV."""
        dlat_m = (target.position.lat - unit.position.lat) * 111_000.0
        dlon_m = (target.position.lon - unit.position.lon) * 111_000.0
        dalt_m = target.position.alt - unit.position.alt
        horiz_m = math.sqrt(dlat_m**2 + dlon_m**2)

        if horiz_m < 1.0:
            return True  # essentially co-located — allow

        # Azimuth offset from shooter heading (North=0, clockwise positive)
        bearing_deg = math.degrees(math.atan2(dlon_m, dlat_m)) % 360.0
        az_diff = (bearing_deg - unit.yaw_deg + 180.0) % 360.0 - 180.0
        if abs(az_diff) > SimplifiedActionProcessor._FIRE_FOV_AZ_DEG:
            return False

        # Elevation offset from shooter pitch
        el_to_target = math.degrees(math.atan2(dalt_m, horiz_m))
        shooter_pitch = getattr(unit, "pitch_deg", 0.0)
        el_diff = (el_to_target - shooter_pitch + 180.0) % 360.0 - 180.0
        return abs(el_diff) <= SimplifiedActionProcessor._FIRE_FOV_EL_DEG

    def _handle_missile_firing(self, unit, trigger_val: float, selected_target, state: dict):
        """Fire a missile via fire_missile_direct() if trigger is active."""
        aircraft_id = getattr(unit, "id", "unknown")

        if not self.trigger_proc.apply_trigger(trigger_val, 3, state):
            return

        if selected_target is None:
            state["vetoed_missile_attempts_this_step"] = 1
            logger.debug(f"Aircraft {aircraft_id}: FIRE_VETO no_target")
            return

        if not self._is_in_fov(unit, selected_target):
            state["vetoed_missile_attempts_this_step"] = 1
            logger.debug(f"Aircraft {aircraft_id}: FIRE_VETO target_outside_fov")
            return

        if not self.weapon_cooldowns.is_missile_ready(state):
            state["vetoed_missile_attempts_this_step"] = 1
            logger.debug(f"Aircraft {aircraft_id}: FIRE_VETO cooldown")
            return

        target_id = getattr(selected_target, "id", None)
        if self.weapon_cooldowns.is_target_saturated(self.simulator, unit.group, target_id):
            state["vetoed_missile_attempts_this_step"] = 1
            logger.debug(f"Aircraft {aircraft_id}: FIRE_VETO target_saturated")
            return

        has_missiles = bool(unit.missile_types)
        if not has_missiles:
            state["vetoed_missile_attempts_this_step"] = 1
            return

        try:
            missile, veto, diagnostics = unit.weapons.fire_missile_direct(
                self.simulator, selected_target, unit.missile_types[0]
            )
        except Exception as exc:
            logger.warning(f"Aircraft {aircraft_id}: fire_missile_direct raised {exc}")
            state["vetoed_missile_attempts_this_step"] = 1
            return

        if missile is not None:
            target_id = diagnostics.get("target_id")
            self.missile_auto.update_missiles_per_target(target_id, True)
            missiles_at_target = self.missile_auto.get_missiles_at_target(target_id)

            if missiles_at_target >= self.missile_auto.max_per_target:
                self.weapon_cooldowns.set_missile_cooldown(state, self.missile_auto.long_cooldown_s)
            else:
                self.weapon_cooldowns.set_missile_cooldown(state)

            state["valid_missile_fires_this_step"] = 1
            logger.info(
                f"Aircraft {aircraft_id}: DIRECT_MISSILE_FIRED target={getattr(selected_target, 'id', 'none')}"
            )
        else:
            state["vetoed_missile_attempts_this_step"] = 1
            logger.debug(f"Aircraft {aircraft_id}: DIRECT_FIRE_VETO {veto}")
