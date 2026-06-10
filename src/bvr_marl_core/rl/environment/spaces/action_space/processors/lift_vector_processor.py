"""
Lift vector command processing.
Converts load factor and bank angle to kinematic rates.
"""

import math

import numpy as np


class LiftVectorProcessor:
    """Process lift-vector commands to attitude control."""

    def __init__(
        self,
        n_min: float = -2.0,
        phi_max_deg: float = 45.0,
        tau_ang: float = 0.25,
        n_neutral: float = 1.0,
    ):
        """
        Initialize lift vector processor.

        Args:
            n_min: Minimum load factor
            phi_max_deg: Maximum bank angle (degrees)
            tau_ang: Time constant for angular rate filtering (seconds)
            n_neutral: Load-factor command at action_n=0.5
        """
        self.n_min = n_min
        self.phi_max_deg = phi_max_deg
        self.tau_ang = tau_ang
        self.n_neutral = n_neutral

    def process_lift_vector_commands(
        self,
        unit,
        action_n: float,
        action_phi: float,
        state: dict,
        dt: float,
        s_factor: float,
        c_factor: float,
    ):
        """
        Process lift-vector commands a^n, a^phi -> attitude commands.

        Centered PPO command convention:
            action_n   = 0.5 -> n_cmd = 1g, the straight-and-level reference
            action_phi = 0.5 -> phi_cmd = 0 deg, wings level

        Values below/above action_n=0.5 unload/push over or pull respectively.
        A banked level turn still needs action_n above 0.5 because vertical lift
        is n * cos(phi).

        Args:
            unit: Aircraft unit
            action_n: Load factor action [0,1], with 0.5 as neutral 1g
            action_phi: Bank angle action [0,1], with 0.5 as wings level
            state: Agent state dict
            dt: Time delta (seconds)
            s_factor: Stall margin scaling factor
            c_factor: Ceiling scaling factor
        """
        # Centered PPO semantics: action_n=0.5 commands 1g instead of
        # the midpoint of [n_min, n_max]. This makes neutral flight easier
        # to discover and easier to reason about from PPO outputs.
        n_max = min(unit.physics.n_max, s_factor * c_factor * unit.physics.n_max)
        n_cmd = self.map_action_to_load_factor(action_n, n_max)

        # Map action_phi to bank angle range; 0.5 is wings level.
        phi_max_scaled = self.phi_max_deg * s_factor * c_factor
        phi_cmd = phi_max_scaled * (2.0 * action_phi - 1.0)

        # Filter commands
        alpha_ang = self._get_alpha_dt_aware(dt, self.tau_ang)
        state["n_cmd_filtered"] = (1 - alpha_ang) * state["n_cmd_filtered"] + alpha_ang * n_cmd
        state["phi_cmd_filtered"] = (1 - alpha_ang) * state[
            "phi_cmd_filtered"
        ] + alpha_ang * phi_cmd

        # Publish commanded n to physics for drag/Ps computation
        try:
            unit.physics.n_external = abs(state["n_cmd_filtered"])
        except Exception:
            pass

        # Convert to kinematic rates and apply
        self._apply_kinematic_rates(unit, state, dt, s_factor, c_factor)

    def map_action_to_load_factor(self, action_n: float, n_max: float) -> float:
        """Map action_n to load factor using this processor's centered convention."""
        return self.action_to_centered_load_factor(action_n, self.n_min, n_max, self.n_neutral)

    def map_load_factor_to_action(self, load_factor: float, n_max: float) -> float:
        """Inverse of map_action_to_load_factor() for scripted controllers."""
        return self.centered_load_factor_to_action(load_factor, self.n_min, n_max, self.n_neutral)

    @staticmethod
    def action_to_centered_load_factor(
        action_n: float,
        n_min: float,
        n_max: float,
        n_neutral: float = 1.0,
    ) -> float:
        """
        Map normalized load-factor action to g command.

        The lower half maps n_min -> n_neutral, and the upper half maps
        n_neutral -> n_max. The neutral value is kept reachable even if the
        live upper envelope is temporarily below it.
        """
        action = float(np.clip(action_n, 0.0, 1.0))
        n_low = min(float(n_min), float(n_neutral))
        n_high = max(float(n_max), float(n_neutral))

        if action <= 0.5:
            return n_neutral + (n_low - n_neutral) * (1.0 - 2.0 * action)
        return n_neutral + (n_high - n_neutral) * (2.0 * action - 1.0)

    @staticmethod
    def centered_load_factor_to_action(
        load_factor: float,
        n_min: float,
        n_max: float,
        n_neutral: float = 1.0,
    ) -> float:
        """
        Inverse of action_to_centered_load_factor() for scripted controllers.

        load_factor=n_neutral maps to 0.5. Commands below neutral map into
        [0, 0.5]; commands above neutral map into [0.5, 1].
        """
        n_cmd = float(load_factor)
        n_low = min(float(n_min), float(n_neutral))
        n_high = max(float(n_max), float(n_neutral))

        if n_cmd < n_neutral and n_neutral > n_low:
            n_cmd = max(n_cmd, n_low)
            return float(
                np.clip(
                    0.5 * (n_cmd - n_low) / (n_neutral - n_low),
                    0.0,
                    0.5,
                )
            )
        if n_cmd > n_neutral and n_high > n_neutral:
            n_cmd = min(n_cmd, n_high)
            return float(
                np.clip(
                    0.5 + 0.5 * (n_cmd - n_neutral) / (n_high - n_neutral),
                    0.5,
                    1.0,
                )
            )
        return 0.5

    def _get_alpha_dt_aware(self, dt: float, tau: float) -> float:
        """Δt-aware exponential smoothing coefficient."""
        return 1.0 - math.exp(-dt / tau)

    def _apply_kinematic_rates(
        self, unit, state: dict, dt: float, s_factor: float, c_factor: float
    ):
        """Convert lift-vector to kinematic rates and integrate."""
        from bvr_marl_core.simulator import normalize_angle

        v = max(unit.speed, 1e-3)
        g = unit.physics.g
        n_filt = state["n_cmd_filtered"]
        phi_filt_rad = math.radians(state["phi_cmd_filtered"])

        # Lift-vector components
        chi_dot_cmd = (g / v) * n_filt * math.sin(phi_filt_rad)  # heading rate
        gamma_dot_cmd = (g / v) * (n_filt * math.cos(phi_filt_rad) - 1.0)  # flight path angle rate

        # Apply rate limits
        omega_max_deg = self._get_max_turn_rate(unit, v, s_factor, c_factor)
        q_max_deg = self._get_max_pitch_rate(unit, s_factor, c_factor)

        chi_dot_cmd = np.clip(math.degrees(chi_dot_cmd), -omega_max_deg, omega_max_deg)
        gamma_dot_cmd = np.clip(math.degrees(gamma_dot_cmd), -q_max_deg, q_max_deg)

        # Convert to body rates and integrate
        delta_yaw = chi_dot_cmd * dt
        new_yaw = normalize_angle(unit.yaw_deg + delta_yaw)
        unit.control.set_yaw_deg(new_yaw)

        delta_pitch = gamma_dot_cmd * dt
        pitch_limit = self._get_pitch_limit(unit)
        new_pitch = np.clip(unit.pitch_deg + delta_pitch, -pitch_limit, pitch_limit)
        # Mirror the physics ground-proximity floor so desired_pitch never fights the blocker.
        # Guard against mocked physics objects where the call returns a non-numeric value.
        if hasattr(unit.physics, "_ground_proximity_pitch_floor"):
            floor_val = unit.physics._ground_proximity_pitch_floor(unit.position.alt)
            if isinstance(floor_val, (int, float)):
                new_pitch = max(new_pitch, floor_val)
        unit.control.set_pitch_deg(new_pitch)

    def _get_max_turn_rate(self, unit, v: float, s_factor: float, c_factor: float) -> float:
        """Get maximum turn rate with envelope scaling."""
        if hasattr(unit.physics, "compute_instantaneous_turn_rate"):
            omega_max_deg = unit.physics.compute_instantaneous_turn_rate(v, unit.position.alt)
        else:
            omega_max_deg = getattr(unit.physics, "max_turn_rate_deg_s", 30.0)
        return omega_max_deg * s_factor * c_factor

    def _get_max_pitch_rate(self, unit, s_factor: float, c_factor: float) -> float:
        """Get maximum pitch rate with envelope scaling."""
        q_max_deg = getattr(unit.physics, "max_pitch_rate_deg_s", 25.0)
        return q_max_deg * s_factor * c_factor

    def _get_pitch_limit(self, unit) -> float:
        """Get pitch angle limit."""
        if hasattr(unit.physics, "get_pitch_limit_deg"):
            return unit.physics.get_pitch_limit_deg(unit.position, unit.speed)
        else:
            return 60.0  # Fallback
