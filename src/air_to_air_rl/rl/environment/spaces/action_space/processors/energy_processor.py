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
        Process energy rate command a^{P_s} -> P_s^{cmd} with asymmetric limits.

        Args:
            unit: Aircraft unit
            action_Ps: Energy rate action [0,1]
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

        # Map action from [0,1] to asymmetric interval [Ps_min, Ps_max]
        Ps_cmd = Ps_min + (Ps_max - Ps_min) * action_Ps

        return Ps_cmd

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
