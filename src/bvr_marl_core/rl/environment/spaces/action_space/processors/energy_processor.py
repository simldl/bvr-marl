"""
Energy command processing.
Converts energy rate actions to throttle settings.
"""


class EnergyProcessor:
    """Process energy rate commands to throttle."""

    def __init__(self, energy_calculator):
        """
        Initialize energy processor.

        Args:
            energy_calculator: EnergyCalculator instance
        """
        self.energy_calculator = energy_calculator

    def process_energy_command(
        self, unit, action_Ps: float, state: dict, dt: float, s_factor: float, c_factor: float
    ) -> float:
        """
        Process energy rate command a^{P_s} -> P_s^{cmd} with centered PPO semantics.

        The normalized policy output is intentionally centered around a neutral
        command:
            action_Ps = 0.0 -> maximum available negative Ps
            action_Ps = 0.5 -> Ps_cmd = 0, hold total specific energy
            action_Ps = 1.0 -> maximum available positive Ps

        This makes the midpoint easy to reason about even when the available
        positive and negative Ps envelopes are asymmetric.

        Args:
            unit: Aircraft unit
            action_Ps: Energy rate action [0,1], with 0.5 as neutral
            state: Agent state dict
            dt: Time delta (seconds)
            s_factor: Stall margin scaling factor
            c_factor: Ceiling scaling factor

        Returns:
            Commanded specific energy rate (m/s)
        """
        # Use filtered values for stability
        alt = unit.position.alt
        v_bar = max(state["v_bar"], 30.0)  # Clamp very low speed
        n_bar = state.get("n_cmd_filtered", 1.0)  # Use filtered load factor

        # Get Ps limits
        Ps_min, Ps_max = self.energy_calculator.get_ps_limits(
            unit, v_bar, alt, n_bar, s_factor, c_factor
        )

        Ps_cmd = self.action_to_centered_ps(action_Ps, Ps_min, Ps_max)

        return Ps_cmd

    @staticmethod
    def action_to_centered_ps(action_Ps: float, Ps_min: float, Ps_max: float) -> float:
        """
        Map normalized energy action to specific excess power.

        The mapping is piecewise so action 0.5 is always Ps=0. The lower half
        spans the available negative Ps range, and the upper half spans the
        available positive Ps range. If one side of the envelope is unavailable,
        that half remains neutral.
        """
        action = max(0.0, min(1.0, float(action_Ps)))
        ps_decel = min(float(Ps_min), 0.0)
        ps_accel = max(float(Ps_max), 0.0)

        if action <= 0.5:
            return ps_decel * (1.0 - 2.0 * action)
        return ps_accel * (2.0 * action - 1.0)

    @staticmethod
    def centered_ps_to_action(Ps_cmd: float, Ps_min: float, Ps_max: float) -> float:
        """
        Inverse of action_to_centered_ps() for scripted controllers.

        Ps_cmd=0 maps to 0.5. Negative commands map into [0, 0.5]; positive
        commands map into [0.5, 1]. Commands beyond the live envelope are
        clipped to the nearest action endpoint.
        """
        ps_cmd = float(Ps_cmd)
        ps_decel = min(float(Ps_min), 0.0)
        ps_accel = max(float(Ps_max), 0.0)

        if ps_cmd < 0.0 and ps_decel < -1e-9:
            ps_cmd = max(ps_cmd, ps_decel)
            return max(0.0, min(0.5, 0.5 * (1.0 - ps_cmd / ps_decel)))
        if ps_cmd > 0.0 and ps_accel > 1e-9:
            ps_cmd = min(ps_cmd, ps_accel)
            return max(0.5, min(1.0, 0.5 + 0.5 * (ps_cmd / ps_accel)))
        return 0.5

    def compute_throttle_from_ps(self, unit, Ps_cmd: float, state: dict, n_factor: float) -> float:
        """
        Compute throttle setting from energy rate command.

        Args:
            unit: Aircraft unit
            Ps_cmd: Commanded specific energy rate
            state: Agent state dict
            n_factor: Load factor

        Returns:
            Throttle setting [0,1]
        """
        import math

        return self.energy_calculator.compute_energy_rate_throttle(
            unit, Ps_cmd, unit.speed, unit.position.alt, math.radians(unit.pitch_deg), n_factor
        )
