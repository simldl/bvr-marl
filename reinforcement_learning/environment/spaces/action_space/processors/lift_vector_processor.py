"""
Lift vector command processing.
Converts load factor and bank angle to kinematic rates.
"""
import math
import numpy as np


class LiftVectorProcessor:
    """Process lift-vector commands to attitude control."""

    def __init__(self, n_min: float = -2.0, phi_max_deg: float = 45.0, tau_ang: float = 0.25):
        """
        Initialize lift vector processor.

        Args:
            n_min: Minimum load factor
            phi_max_deg: Maximum bank angle (degrees)
            tau_ang: Time constant for angular rate filtering (seconds)
        """
        self.n_min = n_min
        self.phi_max_deg = phi_max_deg
        self.tau_ang = tau_ang

    def process_lift_vector_commands(self, unit, action_n: float, action_phi: float, state: dict,
                                     dt: float, s_factor: float, c_factor: float):
        """
        Process lift-vector commands a^{n}, a^{φ} -> attitude commands.

        Args:
            unit: Aircraft unit
            action_n: Load factor action [0,1]
            action_phi: Bank angle action [0,1]
            state: Agent state dict
            dt: Time delta (seconds)
            s_factor: Stall margin scaling factor
            c_factor: Ceiling scaling factor
        """
        # Map action_n to load factor range
        n_max = min(unit.physics.n_max, s_factor * c_factor * unit.physics.n_max)
        n_cmd = self.n_min + (n_max - self.n_min) * action_n

        # Map action_phi to bank angle range
        phi_max_scaled = self.phi_max_deg * s_factor * c_factor
        phi_cmd = phi_max_scaled * (2.0 * action_phi - 1.0)

        # Filter commands
        alpha_ang = self._get_alpha_dt_aware(dt, self.tau_ang)
        state['n_cmd_filtered'] = (1 - alpha_ang) * state['n_cmd_filtered'] + alpha_ang * n_cmd
        state['phi_cmd_filtered'] = (1 - alpha_ang) * state['phi_cmd_filtered'] + alpha_ang * phi_cmd

        # Publish commanded n to physics for drag/Ps computation
        try:
            unit.physics.n_external = abs(state['n_cmd_filtered'])
        except Exception:
            pass

        # Convert to kinematic rates and apply
        self._apply_kinematic_rates(unit, state, dt, s_factor, c_factor)

    def _get_alpha_dt_aware(self, dt: float, tau: float) -> float:
        """Δt-aware exponential smoothing coefficient."""
        return 1.0 - math.exp(-dt / tau)

    def _apply_kinematic_rates(self, unit, state: dict, dt: float, s_factor: float, c_factor: float):
        """Convert lift-vector to kinematic rates and integrate."""
        from simulator.utils.angles import normalize_angle

        v = max(unit.speed, 1e-3)
        g = unit.physics.g
        n_filt = state['n_cmd_filtered']
        phi_filt_rad = math.radians(state['phi_cmd_filtered'])

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
        unit.control.set_pitch_deg(new_pitch)

    def _get_max_turn_rate(self, unit, v: float, s_factor: float, c_factor: float) -> float:
        """Get maximum turn rate with envelope scaling."""
        if hasattr(unit.physics, 'compute_instantaneous_turn_rate'):
            omega_max_deg = unit.physics.compute_instantaneous_turn_rate(v, unit.position.alt)
        else:
            omega_max_deg = getattr(unit.physics, 'max_turn_rate_deg_s', 30.0)
        return omega_max_deg * s_factor * c_factor

    def _get_max_pitch_rate(self, unit, s_factor: float, c_factor: float) -> float:
        """Get maximum pitch rate with envelope scaling."""
        q_max_deg = getattr(unit.physics, 'max_pitch_rate_deg_s', 25.0)
        return q_max_deg * s_factor * c_factor

    def _get_pitch_limit(self, unit) -> float:
        """Get pitch angle limit."""
        if hasattr(unit.physics, 'get_pitch_limit_deg'):
            return unit.physics.get_pitch_limit_deg(unit.position, unit.speed)
        else:
            return 60.0  # Fallback
