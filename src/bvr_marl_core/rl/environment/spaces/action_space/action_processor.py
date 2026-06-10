"""
Main action processor that coordinates all components.
Handles the apply() method and action processing flow.
"""

import warnings

import numpy as np

from .base_processor import ActionProcessorBase
from .weapon_firing import WeaponFiringHandler


class ActionProcessor(ActionProcessorBase):
    """
    Energy + Lift-Vector action processor for BVR air combat.

    Production implementation using energy-based control:
    - Action[0]: Ps - Specific energy rate command; 0.5 = Ps 0 / hold energy
    - Action[1]: n  - Normal load factor command; 0.5 = 1g neutral lift
    - Action[2]: phi - Bank angle command; 0.5 = wings level
    - Actions[3-9]: Target selection, weapons, countermeasures
    """

    def __init__(self, simulator, use_speed_control=None, use_energy_space=None):
        """Initialize action processor."""
        # Legacy parameter warning
        if use_speed_control is not None or (use_energy_space is not None and not use_energy_space):
            warnings.warn(
                "Legacy control modes are deprecated. ActionProcessor now only supports energy-based control.",
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(simulator)
        self.weapon_handler = WeaponFiringHandler(
            self.missile_auto, self.weapon_cooldowns, self.trigger_proc
        )
        self.weapon_handler.simulator = simulator

    def apply(self, agent_id: int, action: np.ndarray):
        """Apply action to aircraft."""
        unit = self.simulator.active_units[agent_id]
        self._init_agent_state(agent_id, unit)
        state = self.agent_states[agent_id]

        # Cleanup and reset
        self.missile_auto.cleanup_destroyed_targets(set(self.simulator.active_units.keys()))
        self.debug_collector.reset_step_counters(state)

        dt = float(getattr(self.simulator, "tick_secs", 1.0))
        action = np.clip(action, 0.0, 1.0)

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
        selected_target = self.target_sorter.select_target(
            unit, self.simulator, action[3], state, dt
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

        alpha_v = self._get_alpha_dt_aware(dt, self.tau_v)
        alpha_ang = self._get_alpha_dt_aware(dt, self.lift_vector_proc.tau_ang)
        alphas = np.array([alpha_v, alpha_ang, alpha_ang] + [0.0] * 7)
        state["u_bar"] = (1 - alphas) * state["u_bar"] + alphas * u_hat
        state["v_bar"] = (1 - alpha_v) * state["v_bar"] + alpha_v * unit.speed

        return u_tilde, u_hat
