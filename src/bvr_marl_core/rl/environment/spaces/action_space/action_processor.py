"""
Main action processor that coordinates all components.
Handles the apply() method and action processing flow.
"""

import warnings

import numpy as np

from bvr_marl_core.domain.information_mode import InformationMode
from bvr_marl_core.rl.environment.spaces.action_space.base_processor import ActionProcessorBase
from bvr_marl_core.rl.environment.spaces.action_space.weapon_firing import WeaponFiringHandler


class ActionProcessor(ActionProcessorBase):
    """
    Energy + Lift-Vector action processor for BVR air combat.

    Production implementation using energy-based control:
    - Action[0]: Ps - Specific energy rate command; 0.5 = Ps 0 / hold energy
    - Action[1]: n  - Normal load factor command; 0.5 = 1g neutral lift
    - Action[2]: phi - Bank angle command; 0.5 = wings level
    - Action[3]: Missile fire
    - Action[4]: Target selection
    - Actions[5-9]: Gun fire and countermeasures
    """

    def __init__(
        self,
        simulator,
        use_speed_control=None,
        use_energy_space=None,
        information_mode=None,
    ):
        """Initialize action processor."""
        # Legacy parameter warning
        if use_speed_control is not None or (use_energy_space is not None and not use_energy_space):
            warnings.warn(
                "Legacy control modes are deprecated. ActionProcessor now only supports energy-based control.",
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(simulator, information_mode=information_mode)
        self.weapon_handler = WeaponFiringHandler(
            self.missile_auto, self.weapon_cooldowns, self.trigger_proc
        )
        self.weapon_handler.simulator = simulator

    def _weapons_in_flight(self, unit):
        """This shooter's own weapons that are still airborne."""
        return self.missile_auto.weapons_in_flight(
            unit, getattr(self.simulator, "active_units", {})
        )

    def apply(self, agent_id: int, action: np.ndarray):
        """Apply action to aircraft."""
        unit = self.simulator.active_units[agent_id]
        self._init_agent_state(agent_id, unit)
        state = self.agent_states[agent_id]

        # Cleanup and reset. The tally is derived from this shooter's weapons that are
        # still in flight, which expires correctly in both selection namespaces.
        self.missile_auto.sync_missiles_in_flight(self._weapons_in_flight(unit))
        self.debug_collector.reset_step_counters(state)

        dt = float(getattr(self.simulator, "tick_secs", 1.0))
        action = np.clip(action, 0.0, 1.0)

        # EMCON radar on/off (action index 9, when present). Silent only in the
        # top quartile so the radar stays ON under any neutral default fill (0.0 or
        # 0.5) — the agent must deliberately drive the channel high to go silent.
        if action.shape[0] > 9:
            unit.radar_emitting = bool(action[9] < 0.75)
        tracker = getattr(unit, "emission_tracker", None)
        # Scripted sensing baseline overrides the policy's radar action (no-op for
        # the default "learned" policy). Applied before recording so the tracker
        # reflects the actually-emitted state.
        emcon = getattr(self, "emcon_controller", None)
        if emcon is not None and emcon.is_override:
            streams = getattr(self.simulator, "random_streams", None)
            rng = streams.generator("emcon", agent_id) if streams is not None else None
            step = tracker.steps if tracker is not None else 0
            forced = emcon.emitting(step=step, unit=unit, rng=rng)
            if forced is not None:
                unit.radar_emitting = bool(forced)
        # Track per-episode emission (duty cycle / toggles) for the active-sensing
        # observation and Pareto analysis. Records the current state every step.
        if tracker is not None:
            tracker.record(getattr(unit, "radar_emitting", True))
            # Persist this unit's duty on the simulator so episode-end metrics cover
            # agents that die mid-episode (the record survives removal from
            # active_units). Keyed by unit id with the team group for filtering.
            records = getattr(self.simulator, "emission_duty_records", None)
            if isinstance(records, dict):
                records[agent_id] = {
                    "group": getattr(unit, "group", None),
                    "duty": tracker.duty_cycle,
                }

        # Process action filtering
        u_tilde, u_hat = self._apply_action_filtering(action, state, dt, unit)

        # Get envelope factors (cached in state so get_envelope_scalars() can reuse without recomputing)
        s_factor, c_factor = self.envelope_calc.compute_envelope_scalars(
            unit, state["v_bar"], unit.position.alt
        )
        state["_cached_s_factor"] = s_factor
        state["_cached_c_factor"] = c_factor

        # Process energy and lift-vector commands
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

        # Target selection
        if self.information_mode is InformationMode.SENSOR_LIMITED:
            selected_target = self.target_sorter.select_contact(
                unit,
                action[4],
                state,
                float(getattr(self.simulator, "elapsed_time_s", 0.0)),
            )
        else:
            selected_target = self.target_sorter.select_target(
                unit, self.simulator, action[4], state, dt
            )

        # Weapon systems
        self.weapon_cooldowns.update_cooldowns(state, dt)
        self.weapon_handler.handle_missile_firing(unit, action, selected_target, state, dt)
        if self.enable_gun:
            self.weapon_handler.handle_gun_firing(unit, action, selected_target, state)
        self.weapon_handler.handle_countermeasures(unit, action, state)

        # Store visualization state
        self.state[unit.id] = {
            "ps_cmd_filtered": Ps_cmd,
            "n_cmd_filtered": state["n_cmd_filtered"],
            "phi_cmd_filtered": state["phi_cmd_filtered"],
            "throttle_filtered": state["throttle_filtered"],
        }

    def _apply_action_filtering(self, action: np.ndarray, state: dict, dt: float, unit) -> tuple:
        """Apply deadzone and exponential filtering to actions."""
        u_tilde = 2.0 * action - 1.0
        u_hat = u_tilde.copy()
        u_hat[1] = self.deadzone_filter.apply_deadzone(u_tilde[1], 1)
        u_hat[2] = self.deadzone_filter.apply_deadzone(u_tilde[2], 2)

        # Only the first three (energy/load/bank) channels are EMA-filtered; every
        # other channel (weapons, countermeasures, EMCON radar toggle) is passed
        # through with zero smoothing. Sizing to len(u_hat) keeps this correct for
        # both the 10-dim and the 11-dim (radar-toggle) action vectors.
        n = len(u_hat)
        u_bar = state["u_bar"]
        if u_bar.shape[0] != n:
            resized = np.zeros(n)
            m = min(n, u_bar.shape[0])
            resized[:m] = u_bar[:m]
            u_bar = resized

        alpha_v = self._get_alpha_dt_aware(dt, self.tau_v)
        alpha_ang = self._get_alpha_dt_aware(dt, self.lift_vector_proc.tau_ang)
        alphas = np.array([alpha_v, alpha_ang, alpha_ang] + [0.0] * (n - 3))
        state["u_bar"] = (1 - alphas) * u_bar + alphas * u_hat
        state["v_bar"] = (1 - alpha_v) * state["v_bar"] + alpha_v * unit.speed

        return u_tilde, u_hat
