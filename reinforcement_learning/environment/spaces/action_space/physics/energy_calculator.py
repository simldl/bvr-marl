"""
Energy rate (specific excess power) calculations.
Maps energy commands to throttle settings.
"""
import numpy as np


class EnergyCalculator:
    """Calculate energy rates and throttle settings."""

    def __init__(self, drag_calculator, eta_P: float = 0.9):
        """
        Initialize energy calculator.

        Args:
            drag_calculator: DragCalculator instance
            eta_P: Safety factor for Ps limits
        """
        self.drag_calculator = drag_calculator
        self.eta_P = eta_P

    def compute_energy_rate_throttle(self, unit, Ps_cmd: float, current_speed: float,
                                     alt: float, pitch_rad: float, n_factor: float) -> float:
        """
        Compute throttle for specific energy rate P_s.

        Args:
            unit: Aircraft unit
            Ps_cmd: Commanded specific energy rate (m/s)
            current_speed: Current velocity (m/s)
            alt: Altitude (meters)
            pitch_rad: Pitch angle (radians)
            n_factor: Load factor (g's)

        Returns:
            Throttle setting [0,1]
        """
        # Clamp very low speeds to avoid division issues
        v_safe = max(current_speed, 30.0)

        # Compute drag D(V,h,n) with load factor
        D_total = self.drag_calculator.compute_drag_with_load_factor(unit, v_safe, alt, n_factor)
        W = unit.physics.mass_kg * unit.physics.g

        # Required thrust: T_req = D + W*P_s/V
        T_req = D_total + W * Ps_cmd / v_safe

        # Convert to throttle setting
        T_max = float(unit.physics.get_engine_force(v_safe, alt, 1.0)) if hasattr(unit.physics, 'get_engine_force') else W
        if T_max > 1e-6:
            throttle = np.clip(T_req / T_max, 0.0, 1.0)
        else:
            throttle = 0.5

        return throttle

    def get_ps_limits(self, unit, v_bar: float, alt: float, n_bar: float, s_factor: float, c_factor: float) -> tuple:
        """
        Get asymmetric P_s limits with envelope scaling.

        Args:
            unit: Aircraft unit
            v_bar: Filtered velocity (m/s)
            alt: Altitude (meters)
            n_bar: Filtered load factor (g's)
            s_factor: Stall margin scaling factor
            c_factor: Ceiling scaling factor

        Returns:
            (Ps_min, Ps_max): Min and max specific energy rates
        """
        # Get thrust bounds at current state
        T_max = unit.physics.get_engine_force(v_bar, alt, 1.0) if hasattr(unit.physics, 'get_engine_force') else 0.0
        T_min = self.drag_calculator.get_min_thrust(unit, v_bar, alt)

        # Compute drag with current load factor D(V,h,n)
        D = self.drag_calculator.compute_drag_with_load_factor(unit, v_bar, alt, n_bar)
        W = unit.physics.mass_kg * unit.physics.g

        # Asymmetric P_s limits with envelope scaling applied to both bounds
        envelope_scale = s_factor * c_factor
        Ps_max = self.eta_P * ((T_max - D) * v_bar / W) * envelope_scale
        Ps_min = self.eta_P * ((T_min - D) * v_bar / W) * envelope_scale

        return Ps_min, Ps_max
